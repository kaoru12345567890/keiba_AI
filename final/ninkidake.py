import pandas as pd
import numpy as np
from tqdm import tqdm # 【ここに付け足し】tqdmをインポート

def calculate_race_performance(file_path):
    df = pd.read_csv(file_path, header=0, low_memory=False)
    
    df.columns = df.columns.str.strip()
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df['人気'] = pd.to_numeric(df['人気'], errors='coerce')
    df = df.dropna(subset=['レースID', '人気', '着順'])
    df = df.sort_values(['レースID', '人気'], ascending=[True, True])
    
    def check_hit(group, n):
        if len(group) < n: return 0
        top_n_horses = group.nsmallest(n, '人気')
        actual_ranks = top_n_horses['着順'].values
        return 1 if (1 in actual_ranks and 2 in actual_ranks and 3 in actual_ranks) else 0

    # レースIDごとのグループを作成
    grouped = df.groupby('レースID')
    
    stats = {}
    
    # 1. 単勝・複勝率
    for i in range(1, 4):
        rank_data = df[df['人気'] == i].set_index('レースID')
        if not rank_data.empty:
            stats[f"{i}st_tanshō"] = (rank_data['着順'] == 1).mean() * 100
            stats[f"{i}st_fukusho"] = (rank_data['着順'] <= 3).mean() * 100

    def is_3rentan(group):
        try:
            return 1 if (group.loc[group['人気']==1, '着順'].values[0] == 1 and
                         group.loc[group['人気']==2, '着順'].values[0] == 2 and
                         group.loc[group['人気']==3, '着順'].values[0] == 3) else 0
        except: return 0

    # 【ここに付け足し】tqdmを使って進捗を表示しながら計算
    # 3連単（固定）
    tqdm.pandas(desc="3連単を計算中") 
    stats["3rentan"] = grouped.progress_apply(is_3rentan, include_groups=False).mean() * 100

    # 【ここに付け足し】3連複系も同様にtqdmで囲む
    # 3連複(3頭BOX)
    tqdm.pandas(desc="3連複を計算中")
    stats["3renpuku"] = grouped.progress_apply(lambda x: check_hit(x, 3), include_groups=False).mean() * 100
    
    # 4頭BOX
    tqdm.pandas(desc="4頭BOXを計算中")
    stats["3renpuku_box4"] = grouped.progress_apply(lambda x: check_hit(x, 4), include_groups=False).mean() * 100
    
    # 5頭BOX
    tqdm.pandas(desc="5頭BOXを計算中")
    stats["3renpuku_box5"] = grouped.progress_apply(lambda x: check_hit(x, 5), include_groups=False).mean() * 100
    
    return stats

# --- 実行 ---
file_path = 'processed_12_data.csv'
performance_results = calculate_race_performance(file_path)

print("=== 評価結果（統一ロジック） ===")
for key, value in performance_results.items():
    print(f"{key:15}: {value:.6f}")