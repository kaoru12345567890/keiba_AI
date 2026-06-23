import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import os

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

# 前処理
place_map = {"札幌": 1, "函館": 2, "福島": 3, "新潟": 4, "東京": 5, "中山": 6, "中京": 7, "京都": 8, "阪神": 9, "小倉": 10}
df['場所'] = df['場所'].map(place_map).fillna(0).astype(int)
for col in ['着順', '馬体重', '体重増減', '斤量', '賞金', '距離', '過去平均着順']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['is_1st'] = (df['着順'] == 1).astype(int)
df['is_top2'] = (df['着順'] <= 2).astype(int)
df['is_top3'] = (df['着順'] <= 3).astype(int)

le_cols = ['天気', '馬場状態', 'レース条件', '芝ダート', '回り', '騎手', '調教師名', '所属', '性別', '曜日']
for col in le_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

exclude_cols = [
    'is_1st', 'is_top2', 'is_top3', 'レースID', 'レース名', '馬名', '馬主', 
    'タイム', '着差', '通過順', '賞金', '性別年齢', '上り3ハロン', '単勝', '人気', '着順',
    '上がり偏差値'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

# 3. 学習と精度出力
target_cols = ['is_1st', 'is_top2', 'is_top3']
params = {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'learning_rate': 0.05, 'max_depth': 6}

train_df = df[df['年'] >= 2025] 

print("--- 学習開始 ---")
for t_col in target_cols:
    # 評価用データを作成して精度を確認
    dataset = lgb.Dataset(train_df[feature_cols].astype(float), train_df[t_col])
    
    # 精度を表示するために eval_set を使う
    model = lgb.train(
        params,
        dataset,
        num_boost_round=500,
        valid_sets=[dataset],
        callbacks=[lgb.log_evaluation(period=0)] # 途中経過を隠して最後にAUCを表示
    )
    
    # 最終的なAUCを取得
    auc_score = model.best_score['training']['auc']
    print(f"モデル '{t_col}' の学習完了 | 精度 (AUC): {auc_score:.4f}")
    
    model.save_model(f'model_{t_col}.txt')