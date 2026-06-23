import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import re

# 1. データ読み込み
df = pd.read_csv('processed_10_data.csv', low_memory=False)

# クレンジング・数値化
numeric_cols = ['年', '月', '日', 'レース目', '出走数', '枠番', '馬番', '斤量', '着順', '人気', '距離', '過去平均着順', '過去出走回数', '過去連対率', '過去複勝率', '過去平均上がり偏差値', '年齢', '馬体重']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(0)) if re.search(r'(\d+\.?\d*)', str(x)) else 0.0)

# 2. 特徴量作成
def add_features(t_df):
    t_df['weight_ratio'] = t_df['馬体重'] / t_df['斤量'].replace(0, 1)
    t_df['weight_diff'] = t_df['馬体重'] - t_df['斤量'] * 8
    t_df['popularity_vs_ability'] = t_df['人気'] / (t_df['過去平均着順'] + 1)
    t_df['weight_age_ratio'] = t_df['馬体重'] * t_df['年齢']
    t_df['レースキー'] = t_df['年'].astype(int).astype(str) + t_df['月'].astype(int).astype(str) + t_df['日'].astype(int).astype(str) + t_df['場所'].astype(str) + t_df['レース目'].astype(int).astype(str)
    for col in ['斤量', '過去平均着順', '人気']:
        if col in t_df.columns:
            t_df[f'{col}_rel'] = t_df.groupby('レースキー')[col].transform(lambda x: x - x.mean())
    return t_df

df = add_features(df)
feat_base = ['人気', '人気_rel', '過去平均着順', '過去複勝率', '斤量', '馬体重', 'weight_ratio', 'weight_diff', 'popularity_vs_ability', 'weight_age_ratio']
years = sorted([y for y in df['年'].unique() if y >= 2021])

# 評価関数（単勝・複勝・3連複）
def evaluate_metrics(df_test, score_col):
    df_test = df_test.copy()
    df_test['rank'] = df_test.groupby('レースキー')[score_col].rank(ascending=False, method='min')
    
    tanshō_hit = df_test.groupby('レースキー').apply(lambda x: (x.loc[x['rank'] == 1, '着順'] == 1).any()).mean()
    fukusho_hit = df_test.groupby('レースキー').apply(lambda x: (x.loc[x['rank'] <= 3, '着順'] <= 3).any()).mean()
    sanrenpuku_hit = df_test.groupby('レースキー').apply(lambda x: (x.loc[x['rank'] <= 3, '着順'] <= 3).sum() == 3).mean()
    
    return tanshō_hit * 100, fukusho_hit * 100, sanrenpuku_hit * 100

all_yearly_results = []

print("--- 年度別・モデル別：単勝・複勝・3連複 詳細評価パイプライン ---")

for target_year in tqdm(years, desc="学習進捗"):
    train = df[df['年'] < target_year].copy()
    test = df[df['年'] == target_year].copy()
    if train.empty or test.empty: continue
    
    # モデル学習
    m_ranker = lgb.LGBMRanker(n_estimators=300).fit(train[feat_base], 19 - train['着順'].clip(upper=18), group=train.groupby('レースキー').size().to_numpy())
    m_class = lgb.LGBMClassifier(n_estimators=300).fit(train[feat_base], (train['着順'] <= 3).astype(int))
    m_reg = lgb.LGBMRegressor(n_estimators=300).fit(train[feat_base], train['着順'])
    
    # 推論と正規化
    scaler = MinMaxScaler()
    test['score_rank'] = scaler.fit_transform(m_ranker.predict(test[feat_base]).reshape(-1, 1))
    test['score_class'] = m_class.predict_proba(test[feat_base])[:, 1]
    test['score_reg'] = scaler.fit_transform((-m_reg.predict(test[feat_base])).reshape(-1, 1))
    
    print(f"\n[{target_year}年 実績]")
    for model_name in ['score_rank', 'score_class', 'score_reg']:
        t, f, s = evaluate_metrics(test, model_name)
        print(f"  - {model_name}: 単勝{t:.1f}% | 複勝{f:.1f}% | 3連複{s:.1f}%")
        
    all_yearly_results.append(test)

# 3. 統合評価
final = pd.concat(all_yearly_results)
final['final_score'] = (final['score_rank'] * 0.5) + (final['score_class'] * 0.3) + (final['score_reg'] * 0.2)
t_final, f_final, s_final = evaluate_metrics(final, 'final_score')

print(f"\n=== 全年度統合モデル 総合評価 ===")
print(f"単勝的中率: {t_final:.2f}%")
print(f"複勝的中率: {f_final:.2f}%")
print(f"3連複的中率: {s_final:.2f}%")