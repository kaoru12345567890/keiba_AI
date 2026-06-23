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
df = df[df['年'] != '年'].copy()
df['年'] = pd.to_numeric(df['年'], errors='coerce').fillna(0).astype(int)

# 2. 前処理
df = df.sort_values(['年', '月', '日', 'レース目']).reset_index(drop=True)
df['レースキー'] = df['年'].astype(str) + df['月'].astype(str) + df['日'].astype(str) + df['場所'] + df['レース目'].astype(str)

# 特徴量リスト
feature_cols = [
    '月', '日', '場所', 'レース目', '天気', '馬場状態', 'レース条件', '芝ダート', '距離', 
    '回り', '出走数', '枠番', '馬番', '斤量', '騎手', '所属', '調教師名', '性別', '年齢', 
    '過去出走回数', '過去平均着順', '過去連対率', '過去複勝率', '過去平均上がり偏差値',
    '馬体重', '体重増減'
]

# 【重要】すべての列を数値化
# カテゴリ系（騎手・調教師など）はLabelEncoderで数値化、それ以外はto_numeric
le_cols = ['天気', '馬場状態', 'レース条件', '芝ダート', '回り', '騎手', '調教師名', '所属', '性別', '曜日', '場所']
for col in le_cols:
    if col in feature_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 相対偏差値の作成（必ず数値化した後に行う）
for col in ['馬体重', '斤量', '過去平均着順']:
    if col in feature_cols:
        df[f'{col}_rel'] = df.groupby('レースキー')[col].transform(lambda x: x - x.mean())
        feature_cols.append(f'{col}_rel')

# 3. ウォークフォワード・ランキング学習
years = sorted([y for y in df['年'].unique() if y >= 2021])
all_results = []

print("--- 学習開始 ---")
for target_year in tqdm(years, desc="学習進捗"):
    train_df = df[df['年'] < target_year]
    test_df = df[df['年'] == target_year].copy()
    
    if test_df.empty: continue
    
    train_groups = train_df.groupby('レースキー', sort=False).size().to_numpy()
    
    # ターゲット: 1着=18, 2着=17, 3着=16, その他=0
    # ここも念のため数値化
    train_df['着順'] = pd.to_numeric(train_df['着順'], errors='coerce').fillna(0)
    y_train = train_df['着順'].apply(lambda x: 18 if x == 1 else (17 if x == 2 else (16 if x == 3 else 0)))
    
    model = lgb.LGBMRanker(objective='lambdarank', n_estimators=300, learning_rate=0.05)
    model.fit(train_df[feature_cols], y_train, group=train_groups)
    
    test_df['score'] = model.predict(test_df[feature_cols])
    all_results.append(test_df)

final_results = pd.concat(all_results)

# 4. 評価
final_results['rank'] = final_results.groupby('レースキー')['score'].rank(ascending=False)
predictions = final_results[final_results['rank'] == 1].copy()
predictions['is_hit'] = (predictions['着順'].astype(int) <= 3).astype(int)

print("\n=== 本命馬の的中率 ===")
print(predictions.groupby('年')['is_hit'].mean())