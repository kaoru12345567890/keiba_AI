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

# 数値変換の徹底
cols_to_numeric = ['着順', '馬体重', '体重増減', '斤量', '賞金', '距離', '過去平均着順']
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ターゲット作成
df['is_1st'] = (df['着順'] == 1).astype(int)
df['is_top2'] = (df['着順'] <= 2).astype(int)
df['is_top3'] = (df['着順'] <= 3).astype(int)

# 函館特化型特徴量
df['is_hakodate_top2'] = ((df['場所'] == 2) & (df['着順'] <= 2)).astype(int)
df['hakodate_dist_feat'] = df['is_hakodate_top2'] * df['距離']

# 3. 特徴量の固定化
drop_cols = ['レースID', 'レース名', '馬名', '馬主', 'タイム', '着差', '通過順', '賞金', 
             '性別年齢', '上り3ハロン', '単勝', '人気', '着順']
df = df.drop(columns=drop_cols, errors='ignore')

# カテゴリ変数処理
le_cols = ['天気', '馬場状態', 'レース条件', '芝ダート', '回り', '騎手', '調教師名', '所属', '性別', '曜日']
for col in le_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 特徴量リストを確定（学習・予測で共通化）
target_cols = ['is_1st', 'is_top2', 'is_top3']
feature_cols = [c for c in df.columns if c not in target_cols]

# 4. 学習処理
lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'learning_rate': 0.05,
    'num_leaves': 31, 'max_depth': 6, 'feature_fraction': 0.7, 'lambda_l2': 0.1
}

years = sorted(df['年'].unique())
all_preds = []

for target_year in tqdm([y for y in years if y >= 2021], desc="学習進捗"):
    train_full = df[df['年'] < target_year].sort_values(['年', '月', '日', 'レース目'])
    test_df = df[df['年'] == target_year].copy()
    
    if test_df.empty: continue
    
    split_idx = int(len(train_full) * 0.8)
    # 明示的に feature_cols で抽出・並び順を固定
    X_train = train_full.iloc[:split_idx][feature_cols].astype(float)
    X_val = train_full.iloc[split_idx:][feature_cols].astype(float)
    X_test = test_df[feature_cols].astype(float)

    for t_col in target_cols:
        model = lgb.train(
            lgb_params,
            lgb.Dataset(X_train, train_full.iloc[:split_idx][t_col]),
            valid_sets=[lgb.Dataset(X_val, train_full.iloc[split_idx:][t_col])],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        test_df[f'pred_{t_col}'] = model.predict(X_test)
    
    all_preds.append(test_df)

final_results = pd.concat(all_preds)
print("\n--- 学習完了 ---")
print(final_results[['pred_is_1st', 'pred_is_top2', 'pred_is_top3']].head(10))