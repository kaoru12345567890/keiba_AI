import pandas as pd
import numpy as np

# ファイルパスの設定
input_file = r'C:\keiba_AI\final\processed_data.csv'
output_file = r'C:\keiba_AI\final\processed_12_data.csv'

print("1. データを読み込み中...")
df = pd.read_csv(input_file, low_memory=False)

# ★ 【作戦実行】「馬名」を守るために「馬名2」をコピーで作る
if '馬名' in df.columns:
    df['馬名2'] = df['馬名']
else:
    raise KeyError("エラー: 元のデータに '馬名' 列がありません。ファイルの破損がないか確認してください。")

# 数字抽出用の関数（数値以外を0に置換、警告対策適用済み）
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
print("2. データを高速集計用に並び替え中...")
df = df.sort_values(by=['馬名2', '年', '月', '日', 'レースID'], ascending=True).reset_index(drop=True)

# クレンジング
clean_order = to_numeric_safe(df['着順'])
clean_deviation = to_numeric_safe(df['上がり偏差値'])

# ==========================================
# 🚀 爆速時系列処理（ベクトル化）
# ==========================================
print("3. 時系列データを爆速集計中（数秒で終わります）...")

# 前走データの作成（馬名2が同じ場合のみ過去データを参照するためのマスク）
is_same_horse_1 = (df['馬名2'] == df['馬名2'].shift(1))
is_same_race_1  = (df['レースID'] == df['レースID'].shift(1))
valid_mask_1 = is_same_horse_1 & (~is_same_race_1)

prev_order = clean_order.shift(1).where(valid_mask_1, np.nan)
prev_dev = clean_deviation.shift(1).where(valid_mask_1, np.nan)

# 直近5戦のローリング計算（groupbyの高速版）
df['過去出走回数'] = prev_order.notna().astype(int).groupby(df['馬名2']).rolling(window=5, min_periods=1).sum().reset_index(drop=True)
df['過去平均着順'] = prev_order.groupby(df['馬名2']).rolling(window=5, min_periods=1).mean().reset_index(drop=True)

# ★ 連対率・複勝率のNaNバグ修正を適用
is_rentai = prev_order.isin([1, 2]).mask(prev_order.isna(), np.nan)
is_fuku = prev_order.isin([1, 2, 3]).mask(prev_order.isna(), np.nan)

df['過去連対率'] = is_rentai.groupby(df['馬名2']).rolling(window=5, min_periods=1).mean().reset_index(drop=True)
df['過去複勝率'] = is_fuku.groupby(df['馬名2']).rolling(window=5, min_periods=1).mean().reset_index(drop=True)
df['過去平均上がり偏差値'] = prev_dev.groupby(df['馬名2']).rolling(window=5, min_periods=1).mean().reset_index(drop=True)

target_cols = ['過去出走回数', '過去平均着順', '過去連対率', '過去複勝率', '過去平均上がり偏差値']
df[target_cols] = df[target_cols].fillna(0)


# ==========================================
# 過去3レース分の情報取得 ＆ 距離差の計算
# ==========================================
shift_target_cols = ['クラス_ランク', '距離', '芝ダート', '着順', '脚質ラベル']

working_df = df.copy()
working_df['着順'] = clean_order

for i in [1, 2, 3]:
    # i走前が同じ馬かつ違うレースIDであるかの判定マスク
    is_same_horse_i = (df['馬名2'] == df['馬名2'].shift(i))
    is_same_race_i  = (df['レースID'] == df['レースID'].shift(i))
    valid_mask_i = is_same_horse_i & (~is_same_race_i)
    
    # 一括シフト
    shifted = working_df[shift_target_cols].shift(i)
    
    # マスク適用（別馬のデータをNaNにする）
    for col in shift_target_cols:
        shifted[col] = shifted[col].where(valid_mask_i, np.nan)
        
    # 新しい列への代入と穴埋め
    for col in shift_target_cols:
        new_col_name = f'{i}走前_{col}'
        if col in ['クラス_ランク', '距離', '着順', '脚質ラベル']:
            df[new_col_name] = shifted[col].fillna(0)
        else:
            df[new_col_name] = shifted[col].fillna('未出走')
            
    # 🆕 今回の距離との「距離差」を一括計算
    is_unraced = (df[f'{i}走前_距離'] == 0)
    df[f'{i}走前_距離差'] = df['距離'] - df[f'{i}走前_距離']
    df.loc[is_unraced, f'{i}走前_距離差'] = 0


# 5. 最終並び替え（番組表順：年、月、日、場所、レース目、着順順）
print("4. データを最終並び替え中...")
df['着順_temp'] = pd.to_numeric(df['着順'], errors='coerce').fillna(99)
df = df.sort_values(by=['年', '月', '日', '場所', 'レース目', '着順_temp']).reset_index(drop=True)

# ★ 【後片付け】一時列と「馬名2」をきれいに削除
drop_cols = ['着順_temp']
if '馬名2' in df.columns:
    drop_cols.append('馬名2')
df = df.drop(columns=drop_cols)

# 6. 保存
print("5. CSVファイルに保存中...")
df.to_csv(output_file, index=False, encoding='utf_8_sig')

print(f"✨ 完了しました！: {output_file} に保存しました。")