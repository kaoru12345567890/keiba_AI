import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

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

# 2. 前処理
place_map = {"札幌": 1, "函館": 2, "福島": 3, "新潟": 4, "東京": 5, "中山": 6, "中京": 7, "京都": 8, "阪神": 9, "小倉": 10}
df['場所'] = df['場所'].map(place_map).fillna(0).astype(int)
for col in ['着順', '馬体重', '体重増減', '斤量', '賞金', '距離', '過去平均着順']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['is_1st'] = (df['着順'] == 1).astype(int)
df['is_hakodate_top2'] = ((df['場所'] == 2) & (df['着順'] <= 2)).astype(int)
df['hakodate_dist_feat'] = df['is_hakodate_top2'] * df['距離']

# 3. 特徴量整理
drop_cols = ['レースID', 'レース名', '馬名', '馬主', 'タイム', '着差', '通過順', '賞金', 
             '性別年齢', '上り3ハロン', '単勝', '人気', '着順', 'is_top2', 'is_top3']
df = df.drop(columns=drop_cols, errors='ignore')

for col in ['天気', '馬場状態', 'レース条件', '芝ダート', '回り', '騎手', '調教師名', '所属', '性別', '曜日']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 4. 学習処理（過学習抑制設定を追加）
years = sorted(df['年'].unique())
all_preds = []

# 過学習を抑えるためのパラメータ
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1,
    'learning_rate': 0.01,         # 学習率を下げて、より慎重に学習させる
    'num_leaves': 31,              # 決定木の葉の数を制限
    'max_depth': 6,                # 木の深さを制限（浅くする）
    'feature_fraction': 0.7,       # 特徴量の70%だけをランダムに使用
    'lambda_l2': 0.1,              # L2正則化で重みの急激な増加を抑制
    'bagging_fraction': 0.7,       # データをランダムにサンプリングして学習
    'bagging_freq': 5
}

for target_year in tqdm([y for y in years if y >= 2021], desc="学習進捗"):
    train_full = df[df['年'] < target_year].sort_values(['年', '月', '日', 'レース目'])
    test_df = df[df['年'] == target_year].copy()
    
    if test_df.empty: continue
        
    split_idx = int(len(train_full) * 0.8)
    train_sub = train_full.iloc[:split_idx].drop(columns=['is_1st']).astype(float)
    val_sub = train_full.iloc[split_idx:].drop(columns=['is_1st']).astype(float)
    
    model = lgb.train(
        lgb_params,
        lgb.Dataset(train_sub, train_full.iloc[:split_idx]['is_1st']),
        valid_sets=[lgb.Dataset(val_sub, train_full.iloc[split_idx:]['is_1st'])],
        num_boost_round=2000, # 多めに設定し、early_stoppingに任せる
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    test_df['pred_is_1st'] = model.predict(test_df.drop(columns=['is_1st']).astype(float))
    all_preds.append(test_df)

final_results = pd.concat(all_preds)
print("\n--- 過学習抑制モデル学習完了 ---")
print(final_results[final_results['場所'] == 2][['pred_is_1st', '距離']].sort_values('pred_is_1st', ascending=False).head(10))