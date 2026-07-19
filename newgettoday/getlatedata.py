import pandas as pd
import numpy as np
import os

# ==============================================================================
# 設定：ファイルパス
# ==============================================================================
ip = input("何日目かを入力してください (例: 01): ")
print(f"指定された日目: 20260201{ip}")
past_file = r'C:\keiba_AI\final\processed_12_data.csv'   # 過去の戦績データ（蓄積ファイル）
new_file = fr'C:\keiba_AI\newgettoday\hakodate_20260201{ip}.csv'  # 新しく作った出馬表ファイル
output_file = fr'C:\keiba_AI\newgettoday\hakodate_2026{ip}.csv' # 特徴量が継ぎ足された新しい出馬表ファイル

print("1. データを読み込んでドッキング中...")
df_past = pd.read_csv(past_file, low_memory=False)
df_new = pd.read_csv(new_file, low_memory=False)

# 後で「元の出馬表の並び順」に完全復元するためのインデックスと、新旧を識別するフラグを追加
df_new['original_index'] = range(len(df_new))
df_new['is_new'] = True
df_past['is_new'] = False

# 新しいデータ（出馬表）に足りない過去データ用の列（着順、上がり偏差値など）を自動作成
# これにより、concatしたときに列がズレずに綺麗に結合できます
for col in df_past.columns:
    if col not in df_new.columns:
        df_new[col] = np.nan

# 過去データ側にも、新データ特有の列があれば自動で合わせる
for col in df_new.columns:
    if col not in df_past.columns:
        df_past[col] = np.nan

# 過去データと新しい出馬表データを縦にドッキング！
df = pd.concat([df_past, df_new], ignore_index=True)

# ★ 【作戦実行】「馬名」を守るために「馬名2」をコピーで作る
if '馬名' in df.columns:
    df['馬名2'] = df['馬名']
else:
    raise KeyError("エラー: 元のデータに '馬名' 列がありません。ファイルの破損がないか確認してください。")

# 数字抽出用の関数（数値以外を0に置換）
def to_numeric_safe(series):
    return pd.to_numeric(series.astype(str).str.extract(r'(\d+)')[0], errors='coerce').fillna(0)

# レース名ランクの判定
df['レース名_str'] = df['レース名'].astype(str).fillna('')
conditions = [
    df['レース名_str'].str.contains('新馬|未勝利'),
    df['レース名_str'].str.contains('500万下|1勝クラス'),
    df['レース名_str'].str.contains('1000万下|2勝クラス'),
    df['レース名_str'].str.contains('1600万下|3勝クラス'),
    df['レース名_str'].str.contains('第') & df['レース名_str'].str.contains('回')
]
choices = [1, 2, 3, 4, 6]
df['クラス_ランク'] = np.select(conditions, choices, default=5)
df.loc[df['レース名'].isna(), 'クラス_ランク'] = 0
df = df.drop(columns=['レース名_str'])

# 2. 並び替え（馬名2ごとに、開催日が古い順に並べる）
# ここで新データ（最新の日付）が必ず各馬の「一番最後」に配置されます！
print("2. データを高速集計用に並び替え中...")
df = df.sort_values(by=['馬名2', '年', '月', '日', 'レースID'], ascending=True).reset_index(drop=True)

# クレンジング（新データの着順は0になりますが、新データ自身が受け取る前走データには影響しません）
clean_order = to_numeric_safe(df['着順'])
clean_deviation = to_numeric_safe(df['上がり偏差値'])

# ==========================================
# 🚀 爆速時系列処理（ベクトル化）
# ==========================================
print("3. 時系列データを爆速集計中...")

# 前走データの作成
is_same_horse_1 = (df['馬名2'] == df['馬名2'].shift(1))
is_same_race_1  = (df['レースID'] == df['レースID'].shift(1))
valid_mask_1 = is_same_horse_1 & (~is_same_race_1)

prev_order = clean_order.shift(1).where(valid_mask_1, np.nan)
prev_dev = clean_deviation.shift(1).where(valid_mask_1, np.nan)

# 直近5戦のローリング計算
df['過去出走回数'] = prev_order.notna().astype(int).groupby(df['馬名2']).rolling(window=5, min_periods=1).sum().reset_index(drop=True)
df['過去平均着順'] = prev_order.groupby(df['馬名2']).rolling(window=5, min_periods=1).mean().reset_index(drop=True)

# 連対率・複勝率の計算
is_rentai = prev_order.isin([1, 2]).mask(prev_order.isna(), np.nan)
is_fuku = prev_order.isin([1, 2, 3]).mask(prev_order.isna(), np.nan)

df['過去連対率'] = is_rentai.groupby(df['馬名2']).rolling(window=5, min_periods=1).mean().reset_index(drop=True)
df['過去複勝率'] = is_fuku.groupby(df['馬名2']).rolling(window=5, min_periods=1).mean().reset_index(drop=True)
df['過去平均上がり偏差値'] = prev_dev.groupby(df['馬名2']).rolling(window=5, min_periods=1).mean().reset_index(drop=True)

target_cols = ['過去出走回数', '過去平均着順', '過去連対率', '過去複勝率', '過去平均上がり偏差値']
df[target_cols] = df[target_cols].fillna(0)

# ==========================================
# 🆕 前走からの間隔（日数・週数）計算
# ==========================================
print("3-2. 前走との間隔を計算中...")
df['今回日付'] = pd.to_datetime(df[['年', '月', '日']].astype(str).agg('-'.join, axis=1), errors='coerce')
prev_date = df['今回日付'].shift(1).where(valid_mask_1, np.nan)

df['前走との間隔_日数'] = (df['今回日付'] - prev_date).dt.days.fillna(0)
df['前走との間隔_週数'] = (df['前走との間隔_日数'] // 7).fillna(0)
df = df.drop(columns=['今回日付'])

df['前走との間隔_日数'] = df['前走との間隔_日数'].astype(int)
df['前走との間隔_週数'] = df['前走との間隔_週数'].astype(int)

# ==========================================
# 過去3レース分の情報取得 ＆ 距離差の計算
# ==========================================
shift_target_cols = ['クラス_ランク', '距離', '芝ダート', '着順', '脚質ラベル']

working_df = df.copy()
working_df['着順'] = clean_order

for i in [1, 2, 3]:
    is_same_horse_i = (df['馬名2'] == df['馬名2'].shift(i))
    is_same_race_i  = (df['レースID'] == df['レースID'].shift(i))
    valid_mask_i = is_same_horse_i & (~is_same_race_i)
    
    shifted = working_df[shift_target_cols].shift(i)
    
    for col in shift_target_cols:
        shifted[col] = shifted[col].where(valid_mask_i, np.nan)
        
    for col in shift_target_cols:
        new_col_name = f'{i}走前_{col}'
        if col in ['クラス_ランク', '距離', '着順', '脚質ラベル']:
            df[new_col_name] = shifted[col].fillna(0)
        else:
            df[new_col_name] = shifted[col].fillna('未出走')
            
    is_unraced = (df[f'{i}走前_距離'] == 0)
    df[f'{i}走前_距離差'] = df['距離'] - df[f'{i}走前_距離']
    df.loc[is_unraced, f'{i}走前_距離差'] = 0

# ==========================================
# 🎯 新しい出馬表データだけを抽出して元の順序に復元
# ==========================================
print("4. 特徴量生成完了！ 不要な結果系カラムおよびスコアを削除中...")

# 新データフラグが True のものだけを綺麗に切り出す
df_shutuba_features = df[df['is_new'] == True].copy()

# スクレイピングした時の「元の出馬表の並び順」に完全復元
df_shutuba_features = df_shutuba_features.sort_values(by='original_index').reset_index(drop=True)

# 管理用・計算用の一時的な列を定義
drop_cols = ['is_new', 'original_index', '馬名2']

# ★不要な結果系カラムに加えて「脚質スコア」を削除対象に追加（※脚質ラベルはここから除外して残します）
unwanted_cols = [
    '着順', '上がり偏差値', '脚質スコア', 
    'タイム', '着差', 'ペース', '通過順', '上り3ハロン', '単勝', '人気'
]

for col in unwanted_cols:
    if col in df_shutuba_features.columns:
        drop_cols.append(col)

# 該当するカラムを一斉に削除
df_shutuba_features = df_shutuba_features.drop(columns=[c for c in drop_cols if c in df_shutuba_features.columns])

# ==========================================
# 📋 カラムの並び順を綺麗に整列
# ==========================================
# ご提示いただいたヘッダーから「脚質スコア」を抜き、「脚質ラベル」を調教師名の次に配置した理想の並び順
final_columns_order = [
    "レースID", "年", "月", "日", "曜日", "場所", "回", "日目", "レース目", "レース名",
    "天気", "馬場状態", "レース条件", "芝ダート", "距離", "回り", "出走数", "枠番",
    "馬番", "馬名", "性別年齢", "斤量", "騎手", "馬体重", "体重増減", "所属", "調教師名",
    "脚質ラベル", "性別", "年齢", "過去出走回数", "過去平均着順", "過去連対率", "過去複勝率",
    "過去平均上がり偏差値", "クラス_ランク", "前走との間隔_日数", "前走との間隔_週数",
    "1走前_クラス_ランク", "1走前_距離", "1走前_芝ダート", "1走前_着順", "1走前_脚質ラベル", "1走前_距離差",
    "2走前_クラス_ランク", "2走前_距離", "2走前_芝ダート", "2走前_着順", "2走前_脚質ラベル", "2走前_距離差",
    "3走前_クラス_ランク", "3走前_距離", "3走前_芝ダート", "3走前_着順", "3走前_脚質ラベル", "3走前_距離差"
]

# 存在する列だけで安全に並び替え
df_shutuba_features = df_shutuba_features.reindex(columns=final_columns_order)

# 6. 保存
print("5. CSVファイルに保存中...")
df_shutuba_features.to_csv(output_file, index=False, encoding='utf_8_sig')

print(f"✨ すべての工程が正常に完了しました！")
print(f"💾 特徴量が整理された最新の出馬表: {output_file}")