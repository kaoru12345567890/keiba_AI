import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import re
import joblib

# ==========================================
# 1. データ読み込みと前処理
# ==========================================
print("--- データの読み込みを開始します ---")
df = pd.read_csv('processed_10_data.csv', low_memory=False)

# 数値化クリーニング処理
numeric_cols = [
    '年', '月', '日', 'レース目', '出走数', '枠番', '馬番', '斤量', '着順', '人気', 
    '距離', '過去平均着順', '過去出走回数', '過去連対率', '過去複勝率', '過去平均上がり偏差値', '年齢', '馬体重'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: float(re.search(r'(\d+\.?\d*)', str(x)).group(0)) if re.search(r'(\d+\.?\d*)', str(x)) else 0.0)

# 【データ最適化】元の「性別」列がすでに数値の場合はそのまま活かし、文字の場合はマッピングします
if '性別' in df.columns:
    # 万が一、文字（牡・牝・セ）で入っていた場合のセーフティマッピング
    if df['性別'].dtype == 'object':
        gender_map = {'牡': 0, '牝': 1, 'セ': 2}
        df['性別_code'] = df['性別'].map(gender_map).fillna(3).astype(int)
    else:
        # すでに数値（2など）が入っている場合は、そのまま整数型としてコピーします
        df['性別_code'] = df['性別'].fillna(3).astype(int)

# ==========================================
# 2. 【新機能】騎手・調教師の自動背番号（ID）マッピング
# ==========================================
print("--- 騎手・調教師の背番号辞書を作成中 ---")
# 文字列の前後にある余計な空白を綺麗に除去して名寄せします
df['騎手'] = df['騎手'].astype(str).str.strip()
df['調教師名'] = df['調教師名'].astype(str).str.strip()

# 重複のないユニークな名前リストを作成し、ソートして0からの背番号を割り振ります
jockey_list = sorted(df['騎手'].unique())
trainer_list = sorted(df['調教師名'].unique())

jockey_dict = {name: idx for idx, name in enumerate(jockey_list)}
trainer_dict = {name: idx for idx, name in enumerate(trainer_list)}

# 元データに背番号（ID）を適用して新しい数値特徴量を作成
df['jockey_id'] = df['騎手'].map(jockey_dict).astype(int)
df['trainer_id'] = df['調教師名'].map(trainer_dict).astype(int)

# ==========================================
# 3. 特徴量生成エンジニアリング
# ==========================================
def add_features(t_df):
    # コースIDの作成
    t_df['course_id'] = t_df['場所'].astype(str) + '_' + t_df['回り'].astype(str)
    t_df['frame_ratio'] = t_df['馬番'] / t_df['出走数'].replace(0, 1)
    
    # 統計用の複合キーの作成
    t_df['course_frame_key'] = t_df['course_id'] + '_' + t_df['馬番'].astype(str)
    t_df['course_style_key'] = (
        t_df['場所'].astype(str) + '_' + 
        t_df['回り'].astype(str) + '_' + 
        t_df['距離'].astype(str) + '_' + 
        t_df['脚質ラベル'].astype(str)
    )
    
    # 馬体重・斤量の物理シナジー
    t_df['weight_ratio'] = t_df['馬体重'] / t_df['斤量'].replace(0, 1)
    t_df['weight_diff'] = t_df['馬体重'] - t_df['斤量'] * 8
    
    # 人気と過去能力のギャップ（歪み）
    t_df['popularity_vs_ability'] = t_df['人気'] / (t_df['過去平均着順'] + 1)
    t_df['weight_age_ratio'] = t_df['馬体重'] * t_df['年齢']
    
    # レースを特定するための固有キー
    t_df['レースキー'] = (
        t_df['年'].astype(int).astype(str) + 
        t_df['月'].astype(int).astype(str) + 
        t_df['日'].astype(int).astype(str) + 
        t_df['場所'].astype(str) + 
        t_df['レース目'].astype(int).astype(str)
    )
    
    # レース内での相対偏差特徴量
    for col in ['斤量', '過去平均着順', '人気']:
        if col in t_df.columns:
            t_df[f'{col}_rel'] = t_df.groupby('レースキー')[col].transform(lambda x: x - x.mean())
    return t_df

print("--- 応用特徴量の生成中 ---")
df = add_features(df)

# 脚質ラベルのワンホットエンコーディング（ダミー変数化）
df = pd.get_dummies(df, columns=['脚質ラベル'], prefix='脚質')

# 【重要】AIモデルに直接投入する特徴量のベースリスト（新設したIDをここに組み込みます）
feat_base = [
    '人気', '人気_rel', '過去平均着順', '過去複勝率', '斤量', '馬体重', 'weight_ratio', 'weight_diff', 
    'popularity_vs_ability', 'weight_age_ratio', 'frame_ratio', '性別_code', 
    'jockey_id', 'trainer_id'  # ◀ 新たに文字から変換された背番号を追加
]
feat_base += [col for col in df.columns if col.startswith('脚質_')]

# 2021年以降をテスト（バックテスト）対象とします
years = sorted([y for y in df['年'].unique() if y >= 2021])

# ==========================================
# 4. 評価関数（バックテスト集計用）
# ==========================================
def evaluate_detailed(df_test, score_col):
    df_test = df_test.copy()
    df_test['rank'] = df_test.groupby('レースキー')[score_col].rank(ascending=False, method='min')
    
    def get_metrics(x):
        top4 = x.loc[x['rank'] <= 4, '着順'].values
        actual_top3 = x.loc[x['着順'] <= 3, '着順'].values
        is_3renpuku_box4 = all([i in top4 for i in actual_top3]) if len(actual_top3) == 3 else False
        
        r1, r2, r3 = x.loc[x['rank'] == 1, '着順'], x.loc[x['rank'] == 2, '着順'], x.loc[x['rank'] == 3, '着順']
        return pd.Series({
            '1st_tanshō': (r1 == 1).any(), '1st_fukusho': (r1 <= 3).any(),
            '2nd_tanshō': (r2 == 1).any(), '2nd_fukusho': (r2 <= 3).any(),
            '3rd_tanshō': (r3 == 1).any(), '3rd_fukusho': (r3 <= 3).any(),
            '3renpuku': (r1 <= 3).any() and (r2 <= 3).any() and (r3 <= 3).any(),
            '3rentan': (r1 == 1).any() and (x.loc[x['rank'] == 2, '着順'] == 2).any() and (x.loc[x['rank'] == 3, '着順'] == 3).any(),
            '3renpuku_box4': is_3renpuku_box4
        })
    return df_test.groupby('レースキー', group_keys=False).apply(get_metrics, include_groups=False).mean() * 100

# ==========================================
# 5. 時系列シミュレーション・学習ループ
# ==========================================
all_results = []
print("--- 騎手・調教師ID＆統合適性スコアを含む学習パイプライン実行中 ---")

# LightGBMに「jockey_id」と「trainer_id」がカテゴリカル変数であることを教えるための指定
categorical_features = ['jockey_id', 'trainer_id']

for target_year in tqdm(years, desc="年度別学習"):
    train = df[df['年'] < target_year].copy()
    test = df[df['年'] == target_year].copy()
    if train.empty or test.empty: continue
    
    # ターゲットエンコーディングによるコース適性統計の算出
    stats_frame_course = train.groupby('course_frame_key')['着順'].apply(lambda x: (x <= 3).mean())
    stats_course_style = train.groupby('course_style_key')['着順'].apply(lambda x: (x <= 3).mean())
    
    for t_df in [train, test]:
        t_df['course_frame_rate'] = t_df['course_frame_key'].map(stats_frame_course).fillna(0)
        t_df['course_style_rate'] = t_df['course_style_key'].map(stats_course_style).fillna(0)
    
    # 最終的な入力特徴量
    feats = feat_base + ['course_frame_rate', 'course_style_rate']
    
    # モデル定義と学習（categorical_feature 引数を追加してパワーアップさせましたわ）
    m_rank = lgb.LGBMRanker(n_estimators=300, random_state=42).fit(
        train[feats], 19 - train['着順'].clip(upper=18), 
        group=train.groupby('レースキー').size().to_numpy(),
        categorical_feature=categorical_features
    )
    m_class = lgb.LGBMClassifier(n_estimators=300, random_state=42).fit(
        train[feats], (train['着順'] <= 3).astype(int),
        categorical_feature=categorical_features
    )
    m_reg = lgb.LGBMRegressor(n_estimators=300, random_state=42).fit(
        train[feats], train['着順'],
        categorical_feature=categorical_features
    )
    
    # スケーリングと予測値算出
    scaler = MinMaxScaler()
    r_p = scaler.fit_transform(m_rank.predict(test[feats]).reshape(-1, 1)).flatten()
    c_p = m_class.predict_proba(test[feats])[:, 1]
    reg_p = scaler.fit_transform((-m_reg.predict(test[feats])).reshape(-1, 1)).flatten()
    
    # アンサンブルモデルのブレンド
    test['Model_A'] = (r_p * 0.4) + (c_p * 0.3) + (reg_p * 0.3)
    test['Model_B'] = r_p
    test['Model_C'] = (c_p * 0.5) + (reg_p * 0.5)
    
    all_results.append(test)

# バックテスト結果の表示
final = pd.concat(all_results)
for model in ['Model_A', 'Model_B', 'Model_C']:
    print(f"\n=== {model} 評価結果 ===")
    for k, v in evaluate_detailed(final, model).items():
        print(f"{k}: {v:.2f}%")

