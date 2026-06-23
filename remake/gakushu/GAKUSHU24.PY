import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import re

# 1. データ読み込み・前処理
df = pd.read_csv('processed_10_data.csv', low_memory=False)
numeric_cols = ['年', '月', '日', 'レース目', '出走数', '枠番', '馬番', '斤量', '着順', '人気', '距離', '過去平均着順', '過去出走回数', '過去連対率', '過去複勝率', '過去平均上がり偏差値', '年齢', '馬体重']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(0)) if re.search(r'(\d+\.?\d*)', str(x)) else 0.0)

def add_features(t_df):
    t_df['weight_ratio'] = t_df['馬体重'] / t_df['斤量'].replace(0, 1)
    t_df['weight_diff'] = t_df['馬体重'] - t_df['斤量'] * 8
    t_df['popularity_vs_ability'] = t_df['人気'] / (t_df['過去平均着順'] + 1)
    t_df['weight_age_ratio'] = t_df['馬体重'] * t_df['年齢']
    t_df['jockey_trainer'] = t_df['騎手'].astype(str) + '_' + t_df['調教師名'].astype(str)
    t_df['レースキー'] = t_df['年'].astype(int).astype(str) + t_df['月'].astype(int).astype(str) + t_df['日'].astype(int).astype(str) + t_df['場所'].astype(str) + t_df['レース目'].astype(int).astype(str)
    for col in ['斤量', '過去平均着順', '人気']:
        if col in t_df.columns:
            t_df[f'{col}_rel'] = t_df.groupby('レースキー')[col].transform(lambda x: x - x.mean())
    return t_df

df = add_features(df)
feat_base = ['人気', '人気_rel', '過去平均着順', '過去複勝率', '斤量', '馬体重', 'weight_ratio', 'weight_diff', 'popularity_vs_ability', 'weight_age_ratio']
years = sorted([y for y in df['年'].unique() if y >= 2021])

# 詳細評価関数
def evaluate_detailed(df_test, score_col):
    df_test = df_test.copy()
    df_test['rank'] = df_test.groupby('レースキー')[score_col].rank(ascending=False, method='min')
    def get_metrics(x):
        r1, r2, r3 = x.loc[x['rank'] == 1, '着順'], x.loc[x['rank'] == 2, '着順'], x.loc[x['rank'] == 3, '着順']
        return pd.Series({
            '1st_tanshō': (r1 == 1).any(), '1st_fukusho': (r1 <= 3).any(),
            '2nd_tanshō': (r2 == 1).any(), '2nd_fukusho': (r2 <= 3).any(),
            '3rd_tanshō': (r3 == 1).any(), '3rd_fukusho': (r3 <= 3).any(),
            '3renpuku': (r1 <= 3).any() and (r2 <= 3).any() and (r3 <= 3).any(),
            '3rentan': (r1 == 1).any() and (x.loc[x['rank'] == 2, '着順'] == 2).any() and (x.loc[x['rank'] == 3, '着順'] == 3).any()
        })
    return df_test.groupby('レースキー', group_keys=False).apply(get_metrics, include_groups=False).mean() * 100

all_results = []
print("--- 3モデル並列評価パイプライン実行中 ---")

for target_year in tqdm(years, desc="年度別学習"):
    train = df[df['年'] < target_year].copy()
    test = df[df['年'] == target_year].copy()
    if train.empty or test.empty: continue
    
    # 統計量
    stats_jockey_cond = train.groupby(['騎手', '場所', '芝ダート'])['着順'].apply(lambda x: (x <= 3).mean())
    stats_trainer_place = train.groupby(['調教師名', '場所'])['着順'].apply(lambda x: (x <= 3).mean())
    stats_jockey_trainer = train.groupby('jockey_trainer')['着順'].apply(lambda x: (x <= 3).mean())
    for t_df in [train, test]:
        t_df['jockey_cond_rate'] = t_df.set_index(['騎手', '場所', '芝ダート']).index.map(stats_jockey_cond).fillna(0)
        t_df['trainer_place_rate'] = t_df.set_index(['調教師名', '場所']).index.map(stats_trainer_place).fillna(0)
        t_df['jockey_trainer_rate'] = t_df.set_index(['jockey_trainer']).index.map(stats_jockey_trainer).fillna(0)
    
    feats = feat_base + ['jockey_cond_rate', 'trainer_place_rate', 'jockey_trainer_rate']
    
    # モデル学習
    m_rank = lgb.LGBMRanker(n_estimators=300).fit(train[feats], 19 - train['着順'].clip(upper=18), group=train.groupby('レースキー').size().to_numpy())
    m_class = lgb.LGBMClassifier(n_estimators=300).fit(train[feats], (train['着順'] <= 3).astype(int))
    m_reg = lgb.LGBMRegressor(n_estimators=300).fit(train[feats], train['着順'])
    
    scaler = MinMaxScaler()
    r_p = scaler.fit_transform(m_rank.predict(test[feats]).reshape(-1, 1)).flatten()
    c_p = m_class.predict_proba(test[feats])[:, 1]
    reg_p = scaler.fit_transform((-m_reg.predict(test[feats])).reshape(-1, 1)).flatten()
    
    # 3つの異なるスコアリング手法
    test['Model_A'] = (r_p * 0.4) + (c_p * 0.3) + (reg_p * 0.3) # 統合型
    test['Model_B'] = r_p                                      # ランキング特化型
    test['Model_C'] = (c_p * 0.5) + (reg_p * 0.5)              # 確率・回帰融合型
    
    all_results.append(test)

final = pd.concat(all_results)
for model in ['Model_A', 'Model_B', 'Model_C']:
    print(f"\n=== {model} 評価結果 ===")
    for k, v in evaluate_detailed(final, model).items():
        print(f"{k}: {v:.2f}%")