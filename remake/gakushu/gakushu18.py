import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re

# 1. データ読み込み
df = pd.read_csv('processed_10_data.csv', low_memory=False)

# 数値化処理
numeric_cols = ['年', '月', '日', 'レース目', '出走数', '枠番', '馬番', '斤量', '着順', '人気', '距離', '過去平均着順', '過去出走回数', '過去連対率', '過去複勝率', '過去平均上がり偏差値', '年齢', '馬体重']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(0)) if re.search(r'(\d+\.?\d*)', str(x)) else 0.0)

# 特徴量生成
df['weight_ratio'] = df['馬体重'] / df['斤量'].replace(0, 1)
df['レースキー'] = df['年'].astype(int).astype(str) + df['月'].astype(int).astype(str) + df['日'].astype(int).astype(str) + df['場所'] + df['レース目'].astype(int).astype(str)

all_results = []
years = sorted([y for y in df['年'].unique() if y >= 2021])

for target_year in tqdm(years, desc="学習進捗"):
    train = df[df['年'] < target_year].copy()
    test = df[df['年'] == target_year].copy()
    if train.empty or test.empty: continue
    
    # 【最重要】学習ラベルを反転させる
    # 着順1着を「18」、2着を「17」...という数値に変える
    # ※最大出走数が18と仮定
    train['y_label'] = 19 - train['着順'].clip(upper=18)
    
    feat = ['過去平均着順', '過去複勝率', '斤量', '馬体重', 'weight_ratio']
    group_size = train.groupby('レースキー').size().to_numpy()
    
    # 反転させたラベルで学習
    model = lgb.LGBMRanker(n_estimators=100, force_col_wise=True)
    model.fit(train[feat], train['y_label'], group=group_size)
    
    # 推論
    test['score'] = model.predict(test[feat])
    test['rank'] = test.groupby('レースキー')['score'].rank(ascending=False, method='min')
    all_results.append(test)

# 2. 精度確認
final_results = pd.concat(all_results)
hit_count = final_results[(final_results['rank'] <= 3) & (final_results['着順'] <= 3)].shape[0]
total_races = final_results['レースキー'].nunique()

print(f"\n=== 的中数: {hit_count} / 総レース数: {total_races} ===")
is_hit = hit_count / (total_races * 3)
print(f"=== 複勝的中率(反転学習版): {is_hit:.2%} ===")