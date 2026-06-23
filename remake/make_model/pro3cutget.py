import pandas as pd
import numpy as np
from tqdm import tqdm

# ファイルパス
input_file = 'processed_3_data.csv'
output_file = 'processed_5_data.csv'

# インデックス定義
IDX_RACE_ID = 0
IDX_NAME = 20  # 馬名のインデックス
IDX_SHUSSOU = 16
IDX_CHAKUJUN = 17
IDX_TSUUKA = 27
IDX_AGARI = 28

# 1. データの読み込み
print("データの読み込み中...")
df = pd.read_csv(input_file, header=None, low_memory=False)

# 2. 異常データ除去
print("異常データのフィルタリング中...")
chaku_col = df[IDX_CHAKUJUN].astype(str)
mask = ~chaku_col.str.contains('除|取|中', na=False)
df_clean = df[mask].copy()

# 3. 上がり3ハロンを数値化（エラー回避）
df_clean[IDX_AGARI] = pd.to_numeric(df_clean[IDX_AGARI], errors='coerce')
# NaNをレース平均で埋める
race_mean_agari = df_clean.groupby(IDX_RACE_ID)[IDX_AGARI].transform('mean')
df_clean[IDX_AGARI] = df_clean[IDX_AGARI].fillna(race_mean_agari)

# 4. 出走頭数再計算
df_clean[IDX_SHUSSOU] = df_clean.groupby(IDX_RACE_ID)[IDX_RACE_ID].transform('count')
race_stats = df_clean.groupby(IDX_RACE_ID)[IDX_AGARI].agg(['mean', 'std'])

# 5. 特徴量計算関数（エラー追跡付き）
def calculate_features(row):
    try:
        n = int(row[IDX_SHUSSOU])
        passing_str = str(row[IDX_TSUUKA])
        
        pos_list = [int(p) for p in passing_str.replace('-', ' ').split()]
        pos_list = [min(p, n) for p in pos_list]
        
        weighted_sum = pos_list[0] if len(pos_list) == 1 else (pos_list[0]*0.6 + pos_list[1]*0.4)
        a = (weighted_sum - 1) / (n - 1) if n > 1 else 0.5
        score = round(a, 3)
        
        label_id = 1 if a < 0.25 else 2 if a < 0.5 else 3 if a < 0.75 else 4
        
        race_id = row[IDX_RACE_ID]
        agari = row[IDX_AGARI]
        stats = race_stats.loc[race_id]
        dev = 50.0 if stats['std'] == 0 else round(50 + 10 * ((stats['mean'] - agari) / stats['std']), 2)
            
        return pd.Series([score, label_id, dev, 0])
    except Exception as e:
        # エラー発生時にIDと馬名を表示
        print(f"\n[エラー検出] レースID: {row[IDX_RACE_ID]}, 馬名: {row[IDX_NAME]}")
        return pd.Series([0.5, 0, 50.0, 1])

# 6. 計算実行
print("計算開始（エラー追跡モード）...")
tqdm.pandas()
results = df_clean.progress_apply(calculate_features, axis=1)

# 結果反映・保存
df_clean['脚質スコア'] = results[0]
df_clean['脚質ラベル'] = results[1]
df_clean['上がり偏差値'] = results[2]
df_clean.to_csv(output_file, index=False, header=False)

print(f"\n処理が完了しました！")
print(f"計算エラー件数: {results[3].sum()}件")