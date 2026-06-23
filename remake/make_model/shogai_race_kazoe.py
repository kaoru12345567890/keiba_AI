import pandas as pd

# ファイルパス
obstacle_file = 'obstacles_data.csv'

# インデックス定義
IDX_RACE_ID = 0

# 1. 障害データの読み込み
df_obs = pd.read_csv(obstacle_file, header=None, low_memory=False)

# 2. レースID単位でユニークな件数をカウント
num_races = df_obs[IDX_RACE_ID].nunique()
num_rows = len(df_obs)

print(f"--- 障害レースの調査結果 ---")
print(f"総データ行数: {num_rows}行")
print(f"障害レース数: {num_races}レース")