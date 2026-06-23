import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import re

# 1. データ読み込み
df = pd.read_csv('processed_10_data.csv', low_memory=False)

# 数値化処理
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

# 詳細評価関数（8項目）
def evaluate_detailed(df_test, score_col):
    df_test = df_test.copy()
    df_test['rank'] = df_test.groupby('レースキー')[score_col].rank(ascending=False, method='min')
    
    def get_metrics(x):
        # 各順位予想馬の着順
        r1 = x.loc[x['rank'] == 1, '着順']
        r2 = x.loc[x['rank'] == 2, '着順']
        r3 = x.loc[x['rank'] == 3, '着順']
        
        # 3連単判定用（rank=1が1着、rank=2が2着、rank=3が3着）
        is_3rentan = (r1 == 1).any() and (x.loc[x['rank'] == 2, '着順'] == 2).any() and (x.loc[x['rank'] == 3, '着順'] == 3).any()
        
        return pd.Series({
            '1st_tanshō': (r1 == 1).any(),
            '1st_fukusho': (r1 <= 3).any(),
            '2nd_tanshō': (r2 == 1).any(),
            '2nd_fukusho': (r2 <= 3).any(),
            '3rd_tanshō': (r3 == 1).any(),
            '3rd_fukusho': (r3 <= 3).any(),
            '3renpuku': (r1 <= 3).any() and (r2 <= 3).any() and (r3 <= 3).any(),
            '3rentan': is_3rentan
        })
    
    return df_test.groupby('レースキー', group_keys=False).apply(get_metrics, include_groups=False).mean() * 100

all_yearly_results = []

print("--- 総合戦略パイプライン実行中 ---")
for target_year in tqdm(years, desc="年度別学習"):
    train = df[df['年'] < target_year].copy()
    test = df[df['年'] == target_year].copy()
    if train.empty or test.empty: continue
    
    m_ranker = lgb.LGBMRanker(n_estimators=300, force_col_wise=True).fit(train[feat_base], 19 - train['着順'].clip(upper=18), group=train.groupby('レースキー').size().to_numpy())
    m_class = lgb.LGBMClassifier(n_estimators=300, force_col_wise=True).fit(train[feat_base], (train['着順'] <= 3).astype(int))
    m_reg = lgb.LGBMRegressor(n_estimators=300, force_col_wise=True).fit(train[feat_base], train['着順'])
    
    scaler = MinMaxScaler()
    test['score_rank'] = scaler.fit_transform(m_ranker.predict(test[feat_base]).reshape(-1, 1))
    test['score_class'] = m_class.predict_proba(test[feat_base])[:, 1]
    test['score_reg'] = scaler.fit_transform((-m_reg.predict(test[feat_base])).reshape(-1, 1))
    
    test['final_score'] = (test['score_rank'] * 0.4) + (test['score_class'] * 0.3) + (test['score_reg'] * 0.3)
    test['score_diff'] = test.groupby('レースキー')['final_score'].transform(lambda x: x.nlargest(2).iloc[0] - x.nlargest(2).iloc[1] if len(x) >= 2 else 0)
    all_yearly_results.append(test)

final = pd.concat(all_yearly_results)

# 結果表示
print("\n=== 全レース統合評価 (8項目) ===")
for k, v in evaluate_detailed(final, 'final_score').items():
    print(f"{k}: {v:.2f}%")

threshold = final['score_diff'].quantile(0.5)
filtered = final[final['score_diff'] >= threshold]
print(f"\n=== 自信度上位50%レースの評価 (8項目) ===")
for k, v in evaluate_detailed(filtered, 'final_score').items():
    print(f"{k}: {v:.2f}%")