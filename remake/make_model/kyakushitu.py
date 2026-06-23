import pandas as pd
import numpy as np

file_path = 'processed_3_data.csv'

# 設定
IDX_RACE_ID, IDX_SHUSSOU, IDX_TSUUKA, IDX_NAME = 0, 16, 27, 20

# 読み込み（必要な列のみ指定してメモリ節約）
df = pd.read_csv(file_path, header=None, low_memory=False, dtype={IDX_RACE_ID: str})

def calculate_score(row):
    passing_order = str(row[IDX_TSUUKA])
    if passing_order == '**' or '-' not in passing_order and not passing_order.isdigit():
        return "判定不能", 0.0

    # 通過順を数値リストに変換
    pos = [int(p) for p in passing_order.replace('-', ' ').split()]
    n = int(row[IDX_SHUSSOU])
    if n <= 1: return "判定不能", 0.0
    
    k = len(pos)
    # 重み付け計算（リストの長さで分岐）
    if k == 4: weighted_sum = pos[0]*0.20 + pos[1]*0.35 + pos[2]*0.25 + pos[3]*0.15
    elif k == 3: weighted_sum = pos[0]*0.35 + pos[1]*0.45 + pos[2]*0.20
    elif k == 2: weighted_sum = pos[0]*0.60 + pos[1]*0.40
    else: weighted_sum = pos[0]
    
    a = (weighted_sum - 1) / (n - 1)
    
    # 判定
    if a < 0.25: label = "逃げ"
    elif a < 0.5: label = "先行"
    elif a < 0.75: label = "差し"
    else: label = "追込"
    return label, round(a, 3)

# データフレーム全体に適用（高速化）
results = df.apply(calculate_score, axis=1, result_type='expand')
df['脚質'], df['スコア'] = results[0], results[1]

# テスト結果を表示
sample = df.sample(n=30)
for _, row in sample.iterrows():
    print(f"ID: {row[IDX_RACE_ID]} | 馬名: {row[IDX_NAME]} | 脚質: {row['脚質']} (スコア: {row['スコア']})")