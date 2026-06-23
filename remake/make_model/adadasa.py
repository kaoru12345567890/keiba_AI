import pandas as pd
import numpy as np
from tqdm import tqdm

# ファイルパスの設定
input_file = 'processed_3_data.csv'
output_file = 'processed_4_data.csv'
error_log_file = 'error_log.txt'

# インデックス定義
IDX_RACE_ID = 0
IDX_SHUSSOU = 16
IDX_NAME = 20
IDX_TSUUKA = 27
IDX_AGARI = 28

# 1. データ読み込み
print("データの読み込みを開始します...")
df = pd.read_csv(input_file, header=None, low_memory=False)
df[IDX_AGARI] = pd.to_numeric(df[IDX_AGARI], errors='coerce')

# 2. 統計値の事前計算
print("レースごとの統計値を計算中...")
race_stats = df.groupby(IDX_RACE_ID)[IDX_AGARI].agg(['mean', 'std'])

# エラーログ用リスト
error_list = []

# 3. 計算関数 (エラー時に情報を記録)
def process_data(row):
    global error_list
    
    # デフォルト値
    score, label_id, dev = 0.5, 0, 50.0
    
    try:
        # --- 脚質計算 ---
        passing_order = str(row[IDX_TSUUKA])
        pos = [int(p) for p in passing_order.replace('-', ' ').split()]
        n = int(row[IDX_SHUSSOU])
        
        if n > 1:
            weighted_sum = pos[0] if len(pos) == 1 else (pos[0]*0.6 + pos[1]*0.4)
            a = (weighted_sum - 1) / (n - 1)
            score = round(a, 3)
            if a < 0.25: label_id = 1
            elif a < 0.5: label_id = 2
            elif a < 0.75: label_id = 3
            else: label_id = 4
            
        # --- 上がり偏差値計算 ---
        race_id = row[IDX_RACE_ID]
        agari = row[IDX_AGARI]
        stats = race_stats.loc[race_id]
        
        if not pd.isna(agari) and stats['std'] > 0:
            dev = round(50 + 10 * ((stats['mean'] - agari) / stats['std']), 2)
            
    except Exception as e:
        # エラー情報を記録
        error_info = f"RaceID: {row[IDX_RACE_ID]}, 馬名: {row[IDX_NAME]}, Error: {str(e)}"
        error_list.append(error_info)
        
    return pd.Series([score, label_id, dev])

# 4. プログレスバー付きで計算
print("特徴量の計算を開始します...")
tqdm.pandas()
results = df.progress_apply(process_data, axis=1)

df['脚質スコア'] = results[0]
df['脚質ラベル'] = results[1]
df['上がり偏差値'] = results[2]

# 5. 結果出力とエラーログ保存
df.to_csv(output_file, index=False, header=False)

if error_list:
    with open(error_log_file, 'w', encoding='utf-8') as f:
        for err in error_list:
            f.write(err + '\n')
    print(f"\n[!] {len(error_list)}件のエラーが発生しました。")
    print(f"詳細は {error_log_file} を確認してください。")
else:
    print("\n[+] エラーなしで完了しました。")

print(f"保存完了: {output_file}")