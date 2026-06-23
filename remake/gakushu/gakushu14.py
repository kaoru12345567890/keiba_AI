import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re

# 1. データの読み込み
df = pd.read_csv('processed_10_data.csv', header=None, low_memory=False)
col_names = [
    'レースID', '年', '月', '日', '曜日', '場所', '回', '日目', 'レース目', 'レース名', 
    '天気', '馬場状態', 'レース条件', '芝ダート', '距離', '回り', '出走数', '着順', 
    '枠番', '馬番', '馬名', '性別年齢', '斤量', '騎手', 'タイム', '着差', 'ペース', 
    '通過順', '上り3ハロン', '単勝', '人気', '馬体重', '体重増減', '所属', '調教師名', '馬主', 
    '賞金', '脚質スコア', '脚質ラベル', '上がり偏差値', '性別', '年齢', '過去出走回数', 
    '過去平均着順', '過去連対率', '過去複勝率', '過去平均上がり偏差値'
]
df.columns = col_names
df = df[df['年'] != '年'].copy()

# クレンジング
def clean_numeric(val):
    val = str(val)
    match = re.search(r'(\d+\.?\d*)', val)
    return float(match.group(0)) if match else 0.0

numeric_cols = ['年', '月', '日', 'レース目', '出走数', '枠番', '馬番', '斤量', '着順', '人気', '距離', '過去平均着順', '過去出走回数', '過去連対率', '過去複勝率', '過去平均上がり偏差値', '年齢']
for col in numeric_cols:
    df[col] = df[col].apply(clean_numeric)

# 前処理
df = df.sort_values(['年', '月', '日', 'レース目']).reset_index(drop=True)
df['レースキー'] = df['年'].astype(int).astype(str) + df['月'].astype(int).astype(str) + df['日'].astype(int).astype(str) + df['場所'] + df['レース目'].astype(int).astype(str)
df['dist_cat'] = pd.cut(df['距離'], bins=[0, 1400, 2200, 9999], labels=[0, 1, 2])

le_cols = ['天気', '馬場状態', 'レース条件', '芝ダート', '回り', '騎手', '調教師名', '所属', '性別', '場所']
for col in le_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 2. 学習ループ
years = sorted([y for y in df['年'].unique() if y >= 2021])
all_results = []

# ベイズ平滑化関数：サンプル不足の統計量を平均に収束させる
def get_smoothed_rate(train_df, group_cols, min_samples=10):
    stats = train_df.groupby(group_cols)['着順'].agg(['mean', 'count'])
    global_mean = (train_df['着順'] <= 3).mean()
    smoothed = (stats['mean'] * stats['count'] + global_mean * min_samples) / (stats['count'] + min_samples)
    return smoothed

print("--- 学習開始：人気重視×統計量平滑化モデル ---")
for target_year in tqdm(years, desc="学習進捗"):
    train_df = df[df['年'] < target_year].copy()
    test_df = df[df['年'] == target_year].copy()
    if train_df.empty or test_df.empty: continue
    
    # 平滑化した統計量を算出
    stats_jockey_cond = get_smoothed_rate(train_df, ['騎手', '場所', '芝ダート'])
    stats_trainer_place = get_smoothed_rate(train_df, ['調教師名', '場所'])
    stats_jockey_dist = get_smoothed_rate(train_df, ['騎手', '場所', 'dist_cat'])
    
    for t_df in [train_df, test_df]:
        t_df.reset_index(drop=True, inplace=True)
        t_df['jockey_cond_rate'] = t_df.set_index(['騎手', '場所', '芝ダート']).index.map(stats_jockey_cond).fillna(0)
        t_df['trainer_place_rate'] = t_df.set_index(['調教師名', '場所']).index.map(stats_trainer_place).fillna(0)
        t_df['jockey_dist_rate'] = t_df.set_index(['騎手', '場所', 'dist_cat']).index.map(stats_jockey_dist).fillna(0)
        
        # 相対偏差値
        for col in ['斤量', '過去平均着順', '人気']:
            t_df[f'{col}_rel'] = t_df.groupby('レースキー')[col].transform(lambda x: x - x.mean())

    # 人気（世間の評価）を最上位に配置し、統計量をスパイスとして活用
    current_features = [
        '人気', '人気_rel', '過去平均着順', '過去複勝率', '斤量', 
        'jockey_cond_rate', 'trainer_place_rate', 'jockey_dist_rate'
    ]
    
    train_x = train_df[current_features].astype(float)
    y_train = train_df['着順'].apply(lambda x: 18 if x == 1 else (17 if x == 2 else (16 if x == 3 else 0)))
    train_groups = train_df.groupby('レースキー', sort=False).size().to_numpy()
    
    # 決定木を増やすことで複雑な人気パターンを学習
    model = lgb.LGBMRanker(objective='lambdarank', n_estimators=500, learning_rate=0.03, num_leaves=31)
    model.fit(train_x, y_train, group=train_groups)
    
    test_df['score'] = model.predict(test_df[current_features].astype(float))
    all_results.append(test_df)

final_results = pd.concat(all_results)
final_results['rank'] = final_results.groupby('レースキー')['score'].rank(ascending=False, method='first')

top3 = final_results[final_results['rank'] <= 3]
is_hit = (top3['着順'] <= 3).mean()
print(f"\n=== モデル完成！ランキング3位以内の複勝的中率: {is_hit:.2%} ===")