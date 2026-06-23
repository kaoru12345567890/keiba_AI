import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score

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

# 2. 競馬場の数値化
place_map = {"札幌": 1, "函館": 2, "福島": 3, "新潟": 4, "東京": 5, "中山": 6, "中京": 7, "京都": 8, "阪神": 9, "小倉": 10}
df['場所'] = df['場所'].map(place_map).fillna(0).astype(int)

# 3. 前処理
# ここで「単勝」と「人気」は数値化して読み込んだ後、学習項目からは除外します
cols_to_numeric = ['着順', '馬体重', '体重増減', '斤量', '賞金']
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 4. 特徴量エンジニアリング
# 函館かどうかを明示的なフラグにする（これが函館適性をAIが意識するトリガーになります）
df['is_hakodate'] = (df['場所'] == 2).astype(int)

# 5. 学習処理から「単勝」「人気」を完全に除外
drop_cols = ['レースID', 'レース名', '馬名', '馬主', 'タイム', '着差', '通過順', '賞金', 
             '性別年齢', '上り3ハロン', '単勝', '人気']
df = df.drop(columns=drop_cols)

le_cols = ['天気', '馬場状態', 'レース条件', '芝ダート', '回り', '騎手', '調教師名', '所属', '性別', '曜日']
for col in le_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

df['is_1st'] = (df['着順'] == 1).astype(int)
df['is_top2'] = (df['着順'] <= 2).astype(int)
df['is_top3'] = (df['着順'] <= 3).astype(int)

# 学習ループはそのまま
years = sorted(df['年'].unique())
test_years = [y for y in years if y >= 2021]
target_cols = ['is_1st', 'is_top2', 'is_top3']
all_preds = []

for target_year in tqdm(test_years, desc="学習進捗"):
    train_df = df[df['年'] < target_year].sort_values(['年', '月', '日', 'レース目'])
    test_df = df[df['年'] == target_year].copy()
    
    if test_df.empty: continue
        
    split_idx = int(len(train_df) * 0.8)
    train_sub = train_df.iloc[:split_idx]
    val_sub = train_df.iloc[split_idx:]
    
    X_test = test_df.drop(columns=['着順', 'is_1st', 'is_top2', 'is_top3'])
    
    for t_col in target_cols:
        model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'learning_rate': 0.05},
            lgb.Dataset(train_sub.drop(columns=['着順', 'is_1st', 'is_top2', 'is_top3']), train_sub[t_col]),
            valid_sets=[lgb.Dataset(val_sub.drop(columns=['着順', 'is_1st', 'is_top2', 'is_top3']), val_sub[t_col])],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        test_df[f'pred_{t_col}'] = model.predict(X_test)
    all_preds.append(test_df)

final_results = pd.concat(all_preds)
print("\n--- 函館適性重視・オッズ除外モデル学習完了 ---")
# 予測結果の確認
print(final_results[['is_hakodate', 'pred_is_1st']].head(10))