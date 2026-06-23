import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re
import optuna
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. データ読み込みと前処理
#-----------------------------------------
#　ベースデータにあり、使える初期要素
# レースID,年,月,日,場所,回,日目,レース目,
# レース名,天気,馬場状態,レース条件,芝ダート,距離,
# 回り,出走数,着順,枠番,馬番,馬名,性別年齢,斤量,騎手,
# タイム,着差,ペース,通過順,上り3ハロン,単勝,人気,馬体重,体重増減,
# 所属,調教師名,馬主,賞金,脚質スコア,脚質ラベル,上がり偏差値,性別,年齢,
# 過去出走回数,過去平均着順,過去連対率,過去複勝率,過去平均上がり偏差値
#-----------------------------------------
#　リークの可能性がある要素
# (着順),上り3ハロン,単勝,人気,上がり偏差値,
#-----------------------------------------
#　今回のモデルで全く使わない要素
# 曜日,着差,通過順,上り3ハロン,上がり偏差値,人気,馬主,賞金,脚質スコア,性別年齢,
#-----------------------------------------
#　今回のモデルで使う要素
# 年,月,日,場所,回,日目,レース目,天気,馬場状態,芝ダート,
# 回り,距離,出走数,枠番,馬番,斤量,着順,人気,馬体重,体重増減,
# 性別,年齢,騎手,所属,調教師名,脚質ラベル,タイム,ペース,単勝,
# 過去出走回数,過去平均着順,過去連対率,過去複勝率,過去平均上がり偏差値
#-----------------------------------------
# 使いたけど使い方がわからない要素
# レースID,レース名,馬名,
# ==========================================

def add_features(t_df):
    t_df['レースキー'] = t_df['年'].astype(int).astype(str) + t_df['月'].astype(int).astype(str) + \
                        t_df['日'].astype(int).astype(str) + t_df['場所'].astype(str) + \
                        t_df['レース目'].astype(int).astype(str)
    return t_df

print("--- データの読み込みと前処理を開始します ---")
df = pd.read_csv('processed_10_data.csv', low_memory=False)

# 数値化処理
numeric_cols = ['年', '月', '日', 'レース目', '出走数', '枠番', '馬番', '斤量', '着順', '距離', '過去出走回数', '過去平均着順', '過去連対率', '過去複勝率', '過去平均上がり偏差値', '年齢', '馬体重']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(0)) if re.search(r'(\d+\.?\d*)', str(x)) else 0.0)

df = add_features(df)

class CategoryEmbedding(nn.Module):
    def __init__(self, num_categories, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_categories, embed_dim)
    def forward(self, x):
        return self.embed(x)

def get_embedding(series, embed_dim=4):
    le = LabelEncoder()
    data = le.fit_transform(series.astype(str))
    model = CategoryEmbedding(len(le.classes_), embed_dim)
    return model(torch.LongTensor(data)).detach().numpy()

print("--- 特徴量のベクトル化を実行中 ---")
embed_features = ['騎手', '調教師名', '馬名', 'レース名']
for col in embed_features:
    embed_data = get_embedding(df[col], embed_dim=4)
    for i in range(4):
        df[f'{col}_e{i}'] = embed_data[:, i]

# ==========================================
# 1. Optunaを用いた最適化と学習
# ==========================================
base_feats = ['斤量', '馬体重', '過去平均着順', '過去連対率', '過去複勝率', '過去平均上がり偏差値', '年齢', '距離']
# embed_featuresの追加も忘れずに
for col in ['騎手', '調教師名', '馬名', 'レース名']:
    base_feats += [f'{col}_e{i}' for i in range(4)]

def objective(trial):
    # 探索するパラメータの範囲
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    
    # 2025年を検証用として使用して最適化
    train = df[df['年'] < 2025].copy()
    val = df[df['年'] == 2025].copy()
    
    y_train = 19 - train['着順'].clip(upper=18)
    
    model = lgb.LGBMRanker(**params, verbose=-1)
    model.fit(
        train[base_feats], 
        y_train, 
        group=train.groupby('レースキー', sort=False).size().to_numpy()
    )
    
    # 検証用データで精度を算出
    val['pred_rank_score'] = model.predict(val[base_feats])
    val['pred_order'] = val.groupby('レースキー', sort=False)['pred_rank_score'].rank(ascending=False, method='first')
    
    # 3連複的中率（4頭BOX）を評価指標にする
    total_races = val['レースキー'].nunique()
    top4 = val[val['pred_order'] <= 4]
    hits = 0
    for _, group in top4.groupby('レースキー'):
        if (group['着順'] <= 3).sum() >= 3:
            hits += 1
            
    return hits / total_races

print("--- ハイパーパラメータの自動探索を開始します ---")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"ベストパラメータ: {study.best_params}")

# ==========================================
# 2. ベストパラメータで全年度学習・結果出力
# ==========================================
print("\n--- ベストモデルによる最終検証実行 ---")
best_params = study.best_params

test_years = range(2021, 2027)
results_list = []

for target_year in test_years:
    train = df[df['年'] < target_year].copy()
    test = df[df['年'] == target_year].copy()
    
    m_rank = lgb.LGBMRanker(**best_params, verbose=-1)
    m_rank.fit(
        train[base_feats], 
        19 - train['着順'].clip(upper=18), 
        group=train.groupby('レースキー', sort=False).size().to_numpy()
    )
    
    # 予測値の算出と代入を追加
    test['pred_rank_score'] = m_rank.predict(test[base_feats])
    test['pred_order'] = test.groupby('レースキー', sort=False)['pred_rank_score'].rank(ascending=False, method='first')
    
    results_list.append(test)

# 【重要】リストを結合して一つのデータフレームにする処理を追加
df_results = pd.concat(results_list)

# ==========================================
# 3. 結果出力
# ==========================================
print("\n" + "="*70)
print(f"{'年度':<6} | {'単勝(1着)':<10} | {'複勝(圏内率)':<12} | {'3連複(4頭BOX)':<14}")
print("-"*70)

for target_year in test_years:
    # 結合した df_results を使用
    test_y = df_results[df_results['年'] == target_year].copy()
    total_races = test_y['レースキー'].nunique()
    
    # 予測ランク上位4頭
    top4 = test_y[test_y['pred_order'] <= 4].copy()
    
    # 1. 単勝：1位予想馬が1着か
    win_hits = top4[(top4['pred_order'] == 1) & (top4['着順'] == 1)].shape[0]
    
    # 2. 複勝：選んだ4頭の中に馬券圏内（3着以内）の馬が何頭いたか
    fukusho_hits = top4[top4['着順'] <= 3].shape[0]
    
    # 3. 3連複(4頭BOX)：選んだ4頭の中に、3着以内の馬が3頭以上含まれているか
    sanrenpuku_hits = 0
    for _, group in top4.groupby('レースキー'):
        if (group['着順'] <= 3).sum() >= 3:
            sanrenpuku_hits += 1
            
    print(f"{target_year:<6} | {win_hits/total_races*100:>8.1f}% | {fukusho_hits/(total_races*4)*100:>10.1f}% | {sanrenpuku_hits/total_races*100:>12.1f}%")

print("="*70)

import matplotlib.pyplot as plt
import seaborn as sns

# 特徴量重要度を取得して可視化する処理
def plot_feature_importance(model, features):
    importances = model.feature_importances_
    feature_imp = pd.DataFrame({'Value': importances, 'Feature': features})
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Feature Importance')
    plt.tight_layout()
    plt.show()

# 最終学習時のモデル(m_rank)を使って表示
# 注意: ループ内の最後のモデルが使われます
plot_feature_importance(m_rank, base_feats)