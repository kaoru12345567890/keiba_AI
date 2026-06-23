import pandas as pd

# ファイルパス
input_file = 'processed_5_data.csv'
output_normal_file = 'processed_6_data.csv'  # 障害を除いたメインデータ
output_obstacle_file = 'obstacles_data.csv'   # 障害データのみ

# インデックス定義
IDX_RACE_TYPE = 13  # 障害レース判定列

# 1. データの読み込み
print("5番データを読み込んでいます...")
df = pd.read_csv(input_file, header=None, low_memory=False)

# 2. 「障」を含む行を抽出
# 文字列として判定し、「障」という文字が含まれている行を探します
is_obstacle = df[IDX_RACE_TYPE].astype(str).str.contains('障', na=False)

df_obstacle = df[is_obstacle].copy()
df_normal = df[~is_obstacle].copy()

# 3. それぞれ別ファイルとして保存
df_obstacle.to_csv(output_obstacle_file, index=False, header=False)
df_normal.to_csv(output_normal_file, index=False, header=False)

print(f"処理が完了しました！")
print(f"障害レースデータ: {len(df_obstacle)}件 -> {output_obstacle_file}")
print(f"平地レースデータ: {len(df_normal)}件 -> {output_normal_file}")