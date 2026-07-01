import pandas as pd
import numpy as np
from tqdm import tqdm

# ファイルパスの設定
input_file = 'processed_data.csv'
output_file = 'processed_12_data.csv'

# 1. データの読み込み
df = pd.read_csv(input_file, low_memory=False)
pd.set_option('future.no_silent_downcasting', True)

# 数字抽出用の関数（数値以外を0に置換）
def to_numeric_safe(series):
    return pd.to_numeric(series.astype(str).str.extract('(\d+)')[0], errors='coerce').fillna(0)

# 2. 並び替え（馬名ごとに、開催日が古い順に並べる）
# 日付情報の列名が '年', '月', '日' である前提です
df = df.sort_values(by=['馬名', '年', '月', '日', 'レースID'], ascending=True)

# 3. グループごとの時系列処理（未来リーク防止付き）
def calc_past_stats(group):
    # groupをコピーして作業用データフレームを作成
    res = group.copy()
    
    # 処理対象列の取得（計算用の数値変換）
    col_order = to_numeric_safe(group['着順'])
    col_deviation = to_numeric_safe(group['上がり偏差値'])
    
    # データを1つ分ずらす（過去のレースを参照）
    prev_order = col_order.shift(1)
    prev_dev = col_deviation.shift(1)
    
    # 【未来リーク防止】同じレースIDの場合は、過去データとして参照しない（NaNにする）
    is_same_race = (group['レースID'] == group['レースID'].shift(1))
    prev_order = prev_order.mask(is_same_race, np.nan)
    prev_dev = prev_dev.mask(is_same_race, np.nan)
    
    # 計算処理（直近5戦の過去実績を新しい列として追加）
    res['過去出走回数'] = prev_order.rolling(window=5, min_periods=1).count()
    res['過去平均着順'] = prev_order.rolling(window=5, min_periods=1).mean()
    res['過去連対率'] = prev_order.isin([1, 2]).rolling(window=5, min_periods=1).mean()
    res['過去複勝率'] = prev_order.isin([1, 2, 3]).rolling(window=5, min_periods=1).mean()
    res['過去平均上がり偏差値'] = prev_dev.rolling(window=5, min_periods=1).mean()
    
    # 新しく作成した列のNaNのみを0で埋める
    target_cols = ['過去出走回数', '過去平均着順', '過去連対率', '過去複勝率', '過去平均上がり偏差値']
    res[target_cols] = res[target_cols].fillna(0)
    
    return res

# 4. 「馬名」でグループ化して適用
grouped = df.groupby('馬名', group_keys=False)
tqdm.pandas(desc="時系列データの集計中")
df_final = grouped.progress_apply(calc_past_stats)

# 5. 並び替え（ご指定の条件）
# リストに指定された列名がデータフレームに存在することを確認してください
df_final['着順_temp'] = pd.to_numeric(df_final['着順'], errors='coerce').fillna(99) # 数値変換、失敗は99（最後尾）へ

df_final = df_final.sort_values(by=['年', '月', '日', '場所', 'レース目', '着順_temp']).reset_index(drop=True)

# 一時的な列を削除
df_final = df_final.drop(columns=['着順_temp'])

# 6. 保存
df_final.to_csv(output_file, index=False, encoding='utf_8_sig')

print(f"完了: {output_file} に保存しました。")