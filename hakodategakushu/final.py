import pandas as pd
import numpy as np
import re
import joblib
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import lightgbm as lgb

# ==========================================
# 警告・ログ非表示設定
# ==========================================
import warnings
import logging
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning) # 念のためパフォーマンス警告も抑制

lgb_logger = logging.getLogger('lightgbm')
lgb_logger.setLevel(logging.ERROR)
for handler in lgb_logger.handlers[:]:
    lgb_logger.removeHandler(handler)
lgb_logger.addHandler(logging.NullHandler())
lgb_logger.propagate = False

lgb_params_default = {'verbose': -1, 'importance_type': 'split'}

MODEL_DIR = r'C:\keiba_AI\final\models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 0. データ読み込み
# ==========================================
print("データ読み込み中...")
df_raw = pd.read_csv(r'C:\keiba_AI\final\processed_12_data.csv', low_memory=False)

print("【条件適用】中央10競馬場すべてのデータを使用します...")
jra_places = '札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉'
df = df_raw[df_raw['場所'].astype(str).str.contains(jra_places, na=False)].copy().reset_index(drop=True)
print(f"全 raw データ: {len(df_raw)}件 -> 中央10競馬場データ: {len(df)}件")

print("数値化処理中...")
numeric_cols = [
    '年', '月', '日', 'レース目', '出走数', '枠番', '馬番', '斤量', '着順', 
    '距離', '単勝', '過去平均着順', '過去出走回数', '過去連対率', '過去複勝率', 
    '過去平均上がり偏差値', '年齢', '馬体重'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

def clean_weight_diff(x):
    x_str = str(x).strip()
    if '計不' in x_str or x_str == 'nan' or x_str == '':
        return np.nan 
    match = re.search(r'([+-]?\d+)', x_str)
    return float(match.group(0)) if match else 0.0

print("馬体重の処理中...")
df['体重増減'] = df['体重増減'].apply(clean_weight_diff).fillna(0.0)

# 【リーク対策】全体平均ではなく固定値で安全に補完
df['馬体重'] = df['馬体重'].fillna(470.0) 
df['過去平均着順'] = df['過去平均着順'].fillna(7.5)

# ==========================================
# 1. 特徴量生成エンジニアリング
# ==========================================
print("クラス_ランクと新馬戦フラグの作成中...")
df['is_新馬戦'] = df['レース名'].str.contains('新馬', na=False).astype(int)

conds = [
    df['レース名'].str.contains('新馬|未勝利', na=False),
    df['レース名'].str.contains('500万下|1勝クラス', na=False),
    df['レース名'].str.contains('1000万下|2勝クラス', na=False),
    df['レース名'].str.contains('1600万下|3勝クラス', na=False),
    df['レース名'].str.contains('第.*回', regex=True)
]
choices = [1, 2, 3, 4, 6]
df['クラス_ランク'] = np.select(conds, choices, default=5)

def apply_race_conditions(t_df):
    cond = t_df['レース条件'].astype(str)
    t_df['斤量_ルール'] = np.select(
        [cond.str.contains('見習'), cond.str.contains('ハンデ'), cond.str.contains('別定')],
        [4, 3, 2], default=1
    )
    t_df['is_牝馬限定'] = cond.str.contains('牝').astype(int)
    t_df['is_ハンデ戦'] = cond.str.contains('ハンデ').astype(int)
    return t_df

df = apply_race_conditions(df)

def add_features(t_df):
    t_df['course_id'] = t_df['場所'].astype(str) + '_' + t_df['回り'].astype(str)
    t_df['frame_ratio'] = t_df['馬番'] / t_df['出走数'].replace(0, 1)
    t_df['course_frame_key'] = t_df['course_id'] + '_' + t_df['馬番'].astype(str)
    t_df['course_style_key'] = (
        t_df['場所'].astype(str) + '_' + t_df['回り'].astype(str) + '_' + 
        t_df['距離'].astype(str) + '_' + t_df['脚質ラベル'].astype(str)
    )
    
    t_df['full_weight'] = t_df['馬体重'] + t_df['斤量']
    t_df['weight_ratio'] = t_df['馬体重'] / t_df['斤量'].replace(0, 1)
    t_df['weight_diff'] = t_df['馬体重'] - t_df['斤量'] * 8
    t_df['popularity_vs_ability'] = (np.log1p(t_df['単勝']) / (t_df['過去平均着順'] + 1))
    t_df['weight_age_ratio'] = t_df['馬体重'] * t_df['年齢']
    t_df['jockey_trainer'] = t_df['騎手'].astype(str) + '_' + t_df['調教師名'].astype(str)
    t_df['レースキー'] = (
        t_df['年'].astype(int).astype(str) + t_df['月'].astype(int).astype(str) + 
        t_df['日'].astype(int).astype(str) + t_df['場所'].astype(str) + t_df['レース目'].astype(int).astype(str)
    )
    
    cols_to_rel = ['weight_ratio', '斤量', '過去平均着順', 'popularity_vs_ability', '年齢', 'full_weight']
    for col in cols_to_rel:
        if col in t_df.columns:
            t_df[f'{col}_rel'] = t_df[col] - t_df.groupby('レースキー')[col].transform('mean')

    for i in [1, 2, 3, 4]:
        col_name_i = f'is_temp_脚質{i}'
        t_df[col_name_i] = (t_df['脚質ラベル'] == i).astype(int)
        t_df[f'脚質{i}_頭数'] = t_df.groupby('レースキー')[col_name_i].transform('sum')
        t_df[f'脚質{i}_割合'] = t_df[f'脚質{i}_頭数'] / t_df['出走数'].replace(0, 1)
        t_df.drop(columns=[col_name_i], inplace=True)
    
    t_df['同型ライバル頭数'] = 0
    for i in [1, 2, 3, 4]:
        mask = (t_df['脚質ラベル'] == i)
        t_df.loc[mask, '同型ライバル頭数'] = t_df.loc[mask, f'脚質{i}_頭数'] - 1
    
    t_df['騎手_競馬場_芝ダート'] = t_df['騎手'].astype(str) + '_' + t_df['場所'] + '_' + t_df['芝ダート']
    t_df['jockey_脚質'] = t_df['騎手'].astype(str) + '_' + t_df['脚質ラベル'].astype(str)
    t_df['場所_脚質'] = t_df['場所'].astype(str) + '_' + t_df['脚質ラベル'].astype(str) + '_' + t_df['芝ダート']
    t_df['馬場状態_脚質'] = t_df['馬場状態'].astype(str) + '_' + t_df['脚質ラベル'].astype(str)
    t_df['場所_芝ダート_馬場状態'] = t_df['場所'].astype(str) + '_' + t_df['芝ダート'].astype(str) + '_' + t_df['馬場状態'].astype(str)
    t_df['天気_脚質'] = t_df['天気'].astype(str) + '_' + t_df['脚質ラベル'].astype(str)
    t_df['天気_騎手'] = t_df['天気'].astype(str) + '_' + t_df['騎手'].astype(str)
    t_df['調教師名_所属'] = t_df['調教師名'].astype(str) + '_' + t_df['所属'].astype(str)
    t_df['調教師名_is_新馬戦'] = t_df['is_新馬戦'].astype(str) + '_' + t_df['調教師名'].astype(str)
    t_df['騎手_is_新馬戦'] = t_df['is_新馬戦'].astype(str) + '_' + t_df['騎手'].astype(str)
    t_df['is_hakodate'] = (t_df['場所'].str.contains('函館')).astype(int)
    t_df['調教師名_is_牝馬限定'] = t_df['is_牝馬限定'].astype(str) + '_' + t_df['調教師名'].astype(str)
    t_df['騎手_is_牝馬限定'] = t_df['is_牝馬限定'].astype(str) + '_' + t_df['騎手'].astype(str)
    t_df['調教師名_is_ハンデ戦'] = t_df['is_ハンデ戦'].astype(str) + '_' + t_df['調教師名'].astype(str)
    t_df['騎手_is_ハンデ戦'] = t_df['is_ハンデ戦'].astype(str) + '_' + t_df['騎手'].astype(str)
    t_df['調教師名_斤量_ルール'] = t_df['斤量_ルール'].astype(str) + '_' + t_df['調教師名'].astype(str)
    t_df['騎手_斤量_ルール'] = t_df['斤量_ルール'].astype(str) + '_' + t_df['騎手'].astype(str)
    t_df['hakodate_jockey_trainer'] = t_df['is_hakodate'].astype(str) + '_' + t_df['jockey_trainer'].astype(str)
    t_df['距離_脚質'] = t_df['距離'].astype(str) + '_' + t_df['脚質ラベル'].astype(str)
    t_df['距離_脚質_馬場状態'] = t_df['距離_脚質'].astype(str) + '_' + t_df['馬場状態'].astype(str)
    
    t_df['is_senba'] = (t_df['性別'] == 1).astype(int)
    t_df['is_female'] = (t_df['性別'] == 2).astype(int)
    t_df['is_male'] = (t_df['性別'] == 3).astype(int)

    if '年齢_rel' in t_df.columns and '斤量_rel' in t_df.columns:
        age_r = t_df['年齢_rel'].astype(str)
        kg_r = t_df['斤量_rel'].astype(str)
        
        t_df['騙馬年齢'] = age_r + '_' + t_df['is_senba'].astype(str)
        t_df['牝馬年齢'] = age_r + '_' + t_df['is_female'].astype(str)
        t_df['牡馬年齢'] = age_r + '_' + t_df['is_male'].astype(str)
        
        t_df['騙馬斤量_rel'] = kg_r + '_' + t_df['騙馬年齢']
        t_df['牝馬斤量_rel'] = kg_r + '_' + t_df['牝馬年齢']
        t_df['牡馬斤量_rel'] = kg_r + '_' + t_df['牡馬年齢']

    return t_df

df = add_features(df)

# 【パフォーマンス対策】大量の列追加によるデータ分断（Fragmented DataFrame）を解消してコピー
df = df.copy()

# ==========================================
# 2. 超・函館札幌特化型の重み付けルール
# ==========================================
print("重み付け処理中...")
max_year = df['年'].max()

def calculate_complex_weight_v3(row):
    weight = 1.0
    place = str(row['場所'])
    
    # カテゴリ数値化されている可能性があるため、文字列変換された値や、元の値に対して判定を安全に行う
    # (LabelEncoder適用前であれば文字列マッチがそのまま有効)
    if '函館' in place or place == '1' or place == '函館': weight *= 10.0
    elif '札幌' in place or place == '2' or place == '札幌': weight *= 8.5
    else: weight *= 1.0
        
    if row['年'] >= (max_year - 2): weight *= 2.0
        
    rank = row['クラス_ランク']
    if rank == 1: weight *= 0.7
    elif rank >= 5: weight *= 1.3
    
    if '稍' in place: weight *= 1.2
    elif '重' in place: weight *= 1.4
    elif '不' in place: weight *= 1.6

    if row.get('Stacking_Score', 0) > 0.8: weight *= 1.5
    return weight

df['weight'] = df.apply(calculate_complex_weight_v3, axis=1)

# ==========================================
# 3. カテゴリ変数のエンコーディング
# ==========================================
print("カテゴリ変数のエンコーディング中...")
categorical_cols = [
    '場所', '回り', "芝ダート", '天気', '馬場状態', '騎手', '所属', 
    '調教師名', 'course_id', 'course_frame_key', 'course_style_key', 'jockey_trainer',
    '騎手_競馬場_芝ダート', 'jockey_脚質', '場所_脚質', '馬場状態_脚質', '場所_芝ダート_馬場状態',
    '天気_脚質', '天気_騎手', '調教師名_所属', '調教師名_is_新馬戦', '騎手_is_新馬戦',
    '調教師名_is_ハンデ戦', '騎手_is_ハンデ戦', 'hakodate_jockey_trainer', '調教師名_is_牝馬限定',
    '騎手_is_牝馬限定', '調教師名_斤量_ルール', '騎手_斤量_ルール', '距離_脚質_馬場状態', '距離_脚質',
    '騙馬年齢', '牝馬年齢', '牡馬年齢', '騙馬斤量_rel', '牝馬斤量_rel', '牡馬斤量_rel',
    '1走前_芝ダート', '2走前_芝ダート', '3走前_芝ダート', '1走前_着順', '2走前_着順', '3走前_着順'
]

encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
joblib.dump(encoders, os.path.join(MODEL_DIR, 'label_encoders.pkl'))

# ==========================================
# 4. 評価関数の定義
# ==========================================
def evaluate_detailed(df_test, score_col):
    df_test = df_test.copy()
    df_test['rank'] = df_test.groupby('レースキー')[score_col].rank(ascending=False, method='min')
    
    def get_metrics(x):
        actual_1st = x.loc[x['着順'] == 1, '着順'].values
        actual_top3 = x.loc[x['着順'] <= 3, '着順'].values
        
        r1 = x.loc[x['rank'] == 1, '着順'].values
        r2 = x.loc[x['rank'] == 2, '着順'].values
        r3 = x.loc[x['rank'] == 3, '着順'].values
        top4 = x.loc[x['rank'] <= 4, '着順'].values
        top5 = x.loc[x['rank'] <= 5, '着順'].values
        
        is_3renpuku_box4 = all([i in top4 for i in actual_top3]) if len(actual_top3) == 3 else False
        is_3rentan_box5 = all([i in top5 for i in actual_top3]) if len(actual_top3) == 3 else False

        return pd.Series({
            '1st_tanshō': (r1 == 1).any(),
            '1st_fukusho': (r1 <= 3).any(),
            '2nd_tanshō': (r2 == 1).any(),
            '2nd_fukusho': (r2 <= 3).any(),
            '3rd_tanshō': (r3 == 1).any(),
            '3rd_fukusho': (r3 <= 3).any(),
            '3renpuku': (r1 <= 3).any() and (r2 <= 3).any() and (r3 <= 3).any(),
            '3rentan': (r1 == 1).any() and (r2 == 2).any() and (r3 == 3).any(),
            '3renpuku_box4': is_3renpuku_box4,
            '3renpuku_box5': is_3rentan_box5
        })
    
    return df_test.groupby('レースキー', group_keys=False).apply(get_metrics, include_groups=False).mean() * 100

# ==========================================
# 5. ターゲットエンコーディング共通関数
# ==========================================
def apply_target_encoding(train_df, test_df, target_cols):
    overall_3rd_rate = (train_df['着順'] <= 3).mean()
    
    def calc_smooth(group):
        return (group.sum() + 10 * overall_3rd_rate) / (len(group) + 10)
        
    res_train = train_df.copy()
    res_test = test_df.copy()
    
    # course_frame_key
    mp = train_df.groupby('course_frame_key')['着順'].apply(lambda x: calc_smooth((x <= 3).astype(int))).to_dict()
    res_train['course_frame_rate'] = res_train['course_frame_key'].map(mp).fillna(overall_3rd_rate)
    res_test['course_frame_rate'] = res_test['course_frame_key'].map(mp).fillna(overall_3rd_rate)
    
    # course_style_key
    mp = train_df.groupby('course_style_key')['着順'].apply(lambda x: calc_smooth((x <= 3).astype(int))).to_dict()
    res_train['course_style_rate'] = res_train['course_style_key'].map(mp).fillna(overall_3rd_rate)
    res_test['course_style_rate'] = res_test['course_style_key'].map(mp).fillna(overall_3rd_rate)
    
    # 騎手_競馬場_芝ダート
    mp = train_df.groupby('騎手_競馬場_芝ダート')['着順'].apply(lambda x: calc_smooth((x <= 3).astype(int))).to_dict()
    res_train['jockey_place_turf_dirt_rate'] = res_train['騎手_競馬場_芝ダート'].map(mp).fillna(overall_3rd_rate)
    res_test['jockey_place_turf_dirt_rate'] = res_test['騎手_競馬場_芝ダート'].map(mp).fillna(overall_3rd_rate)
    
    # 調教師名_場所_芝ダート
    train_df = train_df.copy()
    train_df['trainer_key'] = list(zip(train_df['調教師名'], train_df['場所'], train_df['芝ダート']))
    mp = train_df.groupby('trainer_key')['着順'].apply(lambda x: calc_smooth((x <= 3).astype(int))).to_dict()
    
    res_train['trainer_key'] = list(zip(res_train['調教師名'], res_train['場所'], res_train['芝ダート']))
    res_train['trainer_place_turf_dirt_rate'] = res_train['trainer_key'].map(mp).fillna(overall_3rd_rate)
    res_train.drop(columns=['trainer_key'], inplace=True)
    
    res_test['trainer_key'] = list(zip(res_test['調教師名'], res_test['場所'], res_test['芝ダート']))
    res_test['trainer_place_turf_dirt_rate'] = res_test['trainer_key'].map(mp).fillna(overall_3rd_rate)
    res_test.drop(columns=['trainer_key'], inplace=True)
    
    return res_train, res_test

# ==========================================
# 6. 騎手補正・デバフ用ロジック
# ==========================================
def apply_jockey_boost_v2(local_df, jockey_efficiency_dict, target_col='Model_A'):
    if '騎手' not in local_df.columns: return local_df
    efficiencies = list(jockey_efficiency_dict.values())
    x0 = np.mean(efficiencies) if len(efficiencies) > 0 else 1.0
    k = 0.5 
    def get_sigmoid_boost(eff):
        return 1.0 + ((1 / (1 + np.exp(-k * (eff - x0)))) * 0.2)
    boost_map = {jockey: get_sigmoid_boost(eff) for jockey, eff in jockey_efficiency_dict.items()}
    local_df[target_col] = local_df[target_col] * local_df['騎手'].map(boost_map).fillna(1.0)
    return local_df

def apply_final_correction_v2(local_df, jockey_efficiency_dict, target_col='Model_A'):
    if '騎手' not in local_df.columns: return local_df
    bad_jockeys = [jockey for jockey, eff in jockey_efficiency_dict.items() if eff < 0.3]
    high_perf_jockeys = [jockey for jockey, eff in jockey_efficiency_dict.items() if eff >= 2.0]
    if high_perf_jockeys: local_df.loc[local_df['騎手'].isin(high_perf_jockeys), target_col] *= 1.1
    if bad_jockeys: local_df.loc[local_df['騎手'].isin(bad_jockeys), target_col] *= 0.85
    return local_df

# ==========================================
# 7. レポート関数
# ==========================================
def evaluate_and_print_results(test_df, target_year):
    test_df = test_df.copy()
    try:
        encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
        place_strings = encoders['場所'].inverse_transform(test_df['場所'])
    except:
        place_strings = np.where(test_df['is_hakodate'] == 1, '函館', '他')
    test_df['場所_文字列'] = place_strings
    
    models = ['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Meta_Model']
    print(f"\n==================================================")
    print(f" 📊 【{target_year}年】 中央10競馬場 総合テスト結果 (全 {test_df['レースキー'].nunique()} レース)")
    print(f"==================================================")
    for m in models:
        print(f"--- {m} ---")
        print(evaluate_detailed(test_df, m))
        
    yo_shiba_df = test_df[test_df['場所_文字列'].astype(str).str.contains('函館|札幌', na=False)].copy()
    if not yo_shiba_df.empty:
        print(f"\n==================================================")
        print(f" 🎯 【{target_year}年】 函館・札幌（洋芝開催）のみのテスト結果 (全 {yo_shiba_df['レースキー'].nunique()} レース)")
        print(f"==================================================")
        for m in models:
            print(f"--- {m} ---")
            print(evaluate_detailed(yo_shiba_df, m))

# ==========================================
# 8. バックテスト＆モデリングの実行
# ==========================================
print("モデルの学習開始...")

drop_cols = [
    '着順', 'レースID', 'レースキー', 'レース名', '馬名', 'popularity_vs_ability',
    '年', '月', '日', '曜日', 'レース条件', '脚質ラベル', 'ペース', 'weight', "性別年齢", 
    "タイム", "着差", "通過順", "馬主", "賞金", '回', '日目', 'レース目', '斤量', 
    '上り3ハロン', '単勝', '人気', 'full_weight', '馬体重', '体重増減', '脚質スコア', 
    '上がり偏差値', '天気', '場所', '斤量_ルール', '騎手', 'course_id', '馬場状態', 
    '回り', '枠番', '年齢', '馬番', 'is_ハンデ戦', '性別', 'is_牝馬限定', 'is_新馬戦', 
    '所属', '脚質1_頭数', '脚質2_頭数', '脚質3_頭数', '脚質4_頭数', '芝ダート', 
    'is_hakodate', '距離', 'is_senba', 'is_female', 'is_male', '前走との間隔_週数', 
    '3走前_芝ダート', '2走前_芝ダート', '牡馬斤量_rel', '牝馬斤量_rel'
]
feat_base = [c for c in df.columns if c not in drop_cols]

years = [y for y in sorted(df['年'].unique()) if y >= 2016]

# 【警告対策】代入する前に一旦 df を完全に新調してフラグメンテーションを防ぐ
df = df.copy()
df['Stacking_Score'] = 0.0

lgb_params = {
    'learning_rate': 0.03, 'num_leaves': 31, 'min_child_samples': 20,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'random_state': 42, 'n_jobs': -1, **lgb_params_default
}
meta_context_cols = ['単勝', '出走数', '過去平均着順', 'クラス_ランク']

# ------------------------------------------
# 【仕込みフェーズ】時系列に沿ったプール構築
# ------------------------------------------
print("\n--- 【仕込み期間】2010〜2015年のベースモデル構築中 ---")
init_train = df[(df['年'] >= 2010) & (df['年'] <= 2015)].copy()
meta_features_pool = []
meta_target_pool = []

for stage_y in range(2011, 2016):
    stage_tr = init_train[init_train['年'] < stage_y].copy()
    stage_va = init_train[init_train['年'] == stage_y].copy()
    if stage_tr.empty or stage_va.empty: continue
    
    stage_tr, stage_va = apply_target_encoding(stage_tr, stage_va, [])
    feats_for_init_stacking = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate', 'trainer_place_turf_dirt_rate']

    tmp_rank = lgb.LGBMRanker(n_estimators=150, objective='lambdarank', **lgb_params).fit(
        stage_tr[feats_for_init_stacking], 19 - stage_tr['着順'].clip(upper=18), group=stage_tr.groupby('レースキー').size().to_numpy()
    )
    tmp_class = lgb.LGBMClassifier(n_estimators=150, objective='binary', **lgb_params).fit(
        stage_tr[feats_for_init_stacking], (stage_tr['着順'] <= 3).astype(int)
    )
    tmp_reg = lgb.LGBMRegressor(n_estimators=150, objective='regression', **lgb_params).fit(
        stage_tr[feats_for_init_stacking], stage_tr['着順']
    )
    
    scaler_tmp = MinMaxScaler()
    r_p = scaler_tmp.fit_transform(tmp_rank.predict(stage_va[feats_for_init_stacking]).reshape(-1, 1)).flatten()
    c_p = tmp_class.predict_proba(stage_va[feats_for_init_stacking])[:, 1]
    reg_p = scaler_tmp.fit_transform((-tmp_reg.predict(stage_va[feats_for_init_stacking])).reshape(-1, 1)).flatten()
    
    batch_meta = pd.DataFrame({'r_p': r_p, 'c_p': c_p, 'reg_p': reg_p})
    for ccol in meta_context_cols:
        batch_meta[ccol] = stage_va[ccol].values
        
    meta_features_pool.append(batch_meta)
    meta_target_pool.append((stage_va['着順'] <= 3).astype(int))

# ------------------------------------------
# 【本番バックテスト期間】
# ------------------------------------------
print("\n--- 【本番バックテスト】時系列ローリングループ開始 ---")
for target_year in tqdm(years, desc="本番ローリング実行中"):
    train = df[(df['年'] >= 2010) & (df['年'] < target_year)].copy()
    test = df[df['年'] == target_year].copy()
    if train.empty or test.empty: continue

    X_meta_current_loop = pd.concat(meta_features_pool, ignore_index=True)
    y_meta_current_loop = pd.concat(meta_target_pool, ignore_index=True)
    
    meta_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, 
        random_state=42, objective='binary', **lgb_params_default
    )
    meta_model.fit(X_meta_current_loop, y_meta_current_loop)

    jockey_counts = train['騎手'].value_counts()
    valid_jockeys = jockey_counts[jockey_counts >= 5].index
    jockey_eff_dict = {}
    if len(valid_jockeys) > 0:
        jockey_stats = train[train['騎手'].isin(valid_jockeys)].groupby('騎手').agg({
            '着順': lambda x: (x <= 3).sum(),
            '単勝': lambda x: (1 / x).replace(np.inf, 0).sum()
        })
        jockey_stats['efficiency'] = jockey_stats['着順'] / jockey_stats['単勝'].replace(0, 1)
        jockey_eff_dict = jockey_stats['efficiency'].to_dict()

    train, test = apply_target_encoding(train, test, [])
    feats_for_stacking = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate', 'trainer_place_turf_dirt_rate']
    
    split_idx = int(len(train) * 0.65)
    sub_tr = train.iloc[:split_idx].copy()
    sub_va = train.iloc[split_idx:].copy()
    
    sub_stack_rank = lgb.LGBMRanker(n_estimators=150, objective='lambdarank', **lgb_params).fit(
        sub_tr[feats_for_stacking], 19 - sub_tr['着順'].clip(upper=18), group=sub_tr.groupby('レースキー').size().to_numpy()
    )
    sub_stack_class = lgb.LGBMClassifier(n_estimators=150, objective='binary', **lgb_params).fit(
        sub_tr[feats_for_stacking], (sub_tr['着順'] <= 3).astype(int)
    )
    sub_stack_reg = lgb.LGBMRegressor(n_estimators=150, objective='regression', **lgb_params).fit(
        sub_tr[feats_for_stacking], sub_tr['着順']
    )
    
    scaler_test = MinMaxScaler()
    sub_va_r = scaler_test.fit_transform(sub_stack_rank.predict(sub_va[feats_for_stacking]).reshape(-1, 1)).flatten()
    sub_va_c = sub_stack_class.predict_proba(sub_va[feats_for_stacking])[:, 1]
    sub_va_reg = scaler_test.fit_transform((-sub_stack_reg.predict(sub_va[feats_for_stacking])).reshape(-1, 1)).flatten()
    
    sub_va_batch = pd.DataFrame({'r_p': sub_va_r, 'c_p': sub_va_c, 'reg_p': sub_va_reg})
    for ccol in meta_context_cols: sub_va_batch[ccol] = sub_va[ccol].values
    
    stack_rank = lgb.LGBMRanker(n_estimators=300, objective='lambdarank', **lgb_params).fit(
        train[feats_for_stacking], 19 - train['着順'].clip(upper=18), group=train.groupby('レースキー').size().to_numpy()
    )
    stack_class = lgb.LGBMClassifier(n_estimators=300, objective='binary', **lgb_params).fit(
        train[feats_for_stacking], (train['着順'] <= 3).astype(int)
    )
    stack_reg = lgb.LGBMRegressor(n_estimators=300, objective='regression', **lgb_params).fit(
        train[feats_for_stacking], train['着順']
    )
    
    test_r_p = scaler_test.fit_transform(stack_rank.predict(test[feats_for_stacking]).reshape(-1, 1)).flatten()
    test_c_p = stack_class.predict_proba(test[feats_for_stacking])[:, 1]
    test_reg_p = scaler_test.fit_transform((-stack_reg.predict(test[feats_for_stacking])).reshape(-1, 1)).flatten()
    
    test_batch_df = pd.DataFrame({'r_p': test_r_p, 'c_p': test_c_p, 'reg_p': test_reg_p})
    for ccol in meta_context_cols: test_batch_df[ccol] = test[ccol].values
    
    test['Stacking_Score'] = meta_model.predict_proba(test_batch_df)[:, 1]
    df.loc[test.index, 'Stacking_Score'] = test['Stacking_Score']

    train['Stacking_Score'] = 0.0
    train.iloc[split_idx:, train.columns.get_loc('Stacking_Score')] = meta_model.predict_proba(sub_va_batch)[:, 1]

    current_feats = feats_for_stacking + ['Stacking_Score']
    train_weights = train.apply(calculate_complex_weight_v3, axis=1).values
    
    val_size = int(len(train) * 0.15)
    train_idx, val_idx = train.index[:-val_size], train.index[-val_size:]
    
    X_tr, y_tr = train.loc[train_idx, current_feats], train.loc[train_idx]
    X_val, y_val = train.loc[val_idx, current_feats], train.loc[val_idx]
    w_tr, w_val = train_weights[:-val_size], train_weights[-val_size:]

    m_rank = lgb.LGBMRanker(n_estimators=1000, objective='lambdarank', **lgb_params)
    m_rank.fit(
        X_tr, 19 - y_tr['着順'].clip(upper=18), group=y_tr.groupby('レースキー').size().to_numpy(), sample_weight=w_tr,
        eval_set=[(X_val, 19 - y_val['着順'].clip(upper=18))], eval_group=[y_val.groupby('レースキー').size().to_numpy()],
        eval_sample_weight=[w_val], callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    m_class = lgb.LGBMClassifier(n_estimators=1000, objective='binary', **lgb_params)
    m_class.fit(
        X_tr, (y_tr['着順'] <= 3).astype(int), sample_weight=w_tr,
        eval_set=[(X_val, (y_val['着順'] <= 3).astype(int))], eval_sample_weight=[w_val],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    m_reg = lgb.LGBMRegressor(n_estimators=1000, objective='regression', **lgb_params)
    m_reg.fit(
        X_tr, y_tr['着順'], sample_weight=w_tr,
        eval_set=[(X_val, y_val['着順'])], eval_sample_weight=[w_val],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    r_p_final = scaler_test.fit_transform(m_rank.predict(test[current_feats]).reshape(-1, 1)).flatten()
    c_p_final = m_class.predict_proba(test[current_feats])[:, 1]
    reg_p_final = scaler_test.fit_transform((-m_reg.predict(test[current_feats])).reshape(-1, 1)).flatten()
    
    test['Model_A'] = (r_p_final * 0.4) + (c_p_final * 0.3) + (reg_p_final * 0.3)
    test['Model_B'] = (r_p_final * 0.2) + (c_p_final * 0.6) + (reg_p_final * 0.2)
    test['Model_C'] = (c_p_final * 0.5) + (reg_p_final * 0.5)
    test['Model_D'] = (c_p_final)
    test['Meta_Model'] = test['Stacking_Score']
    
    for m in ['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Meta_Model']:
        test = apply_jockey_boost_v2(test, jockey_eff_dict, target_col=m)
        test = apply_final_correction_v2(test, jockey_eff_dict, target_col=m)

    evaluate_and_print_results(test, target_year)
    meta_features_pool.append(test_batch_df)
    meta_target_pool.append((test['着順'] <= 3).astype(int))

# ==========================================
# 9. メタモデルの最終決定と保存
# ==========================================
print("\n" + "="*50)
print(" 🌀 メタモデル（スタッキング）の最終学習を開始...")
print("="*50)
X_meta = pd.concat(meta_features_pool, ignore_index=True)
y_meta = pd.concat(meta_target_pool, ignore_index=True)

meta_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, random_state=42, objective='binary', **lgb_params_default)
meta_model.fit(X_meta, y_meta)
joblib.dump(meta_model, os.path.join(MODEL_DIR, 'meta_model.pkl'))
print("-> [保存成功] meta_model.pkl (スタッキング・メタモデル)")

# ==========================================
# 10. 【120%完全版】全アセットの一括保存処理（追加セクション）
# ==========================================
print("\n" + "="*50)
print(" 💾 【完全防衛線】推論実力を120%出すための全アセット保存フェーズへ突入します")
print("="*50)

# ① ターゲットエンコーディング（TE）最終マップの抽出と保存
print("-> 全データを用いたターゲットエンコーディング最終マップを構築中...")
overall_3rd_rate = (df['着順'] <= 3).mean()
def calc_smooth_final(group):
    return (group.sum() + 10 * overall_3rd_rate) / (len(group) + 10)

te_maps = {
    'overall_3rd_rate': overall_3rd_rate,
    'course_frame_key': df.groupby('course_frame_key')['着順'].apply(lambda x: calc_smooth_final((x <= 3).astype(int))).to_dict(),
    'course_style_key': df.groupby('course_style_key')['着順'].apply(lambda x: calc_smooth_final((x <= 3).astype(int))).to_dict(),
    '騎手_競馬場_芝ダート': df.groupby('騎手_競馬場_芝ダート')['着順'].apply(lambda x: calc_smooth_final((x <= 3).astype(int))).to_dict(),
}
df['trainer_key'] = list(zip(df['調教師名'], df['場所'], df['芝ダート']))
te_maps['trainer_key'] = df.groupby('trainer_key')['着順'].apply(lambda x: calc_smooth_final((x <= 3).astype(int))).to_dict()
joblib.dump(te_maps, os.path.join(MODEL_DIR, 'te_maps.pkl'))
print("-> [保存成功] te_maps.pkl (ターゲットエンコーディングマップ)")

# ② 騎手エフィシエンシーの全期間集計と保存
print("-> 全データを用いた騎手成績エフィシエンシーの抽出中...")
jockey_counts = df['騎手'].value_counts()
valid_jockeys = jockey_counts[jockey_counts >= 5].index
jockey_eff_dict_final = {}
if len(valid_jockeys) > 0:
    jockey_stats = df[df['騎手'].isin(valid_jockeys)].groupby('騎手').agg({
        '着順': lambda x: (x <= 3).sum(),
        '単勝': lambda x: (1 / x).replace(np.inf, 0).sum()
    })
    jockey_stats['efficiency'] = jockey_stats['着順'] / jockey_stats['単勝'].replace(0, 1)
    jockey_eff_dict_final = jockey_stats['efficiency'].to_dict()
joblib.dump(jockey_eff_dict_final, os.path.join(MODEL_DIR, 'jockey_eff_dict.pkl'))
print("-> [保存成功] jockey_eff_dict.pkl (騎手デバフマップ)")

# ③ 全データを使用した「子モデル（ベースモデル3種）」の最終フィッティング
print("-> 全データを用いた最終ベースモデル（Rank/Class/Reg）の全力学習中...")
df_encoded = df.copy()
df_encoded['course_frame_rate'] = df_encoded['course_frame_key'].map(te_maps['course_frame_key']).fillna(overall_3rd_rate)
df_encoded['course_style_rate'] = df_encoded['course_style_key'].map(te_maps['course_style_key']).fillna(overall_3rd_rate)
df_encoded['jockey_place_turf_dirt_rate'] = df_encoded['騎手_競馬場_芝ダート'].map(te_maps['騎手_競馬場_芝ダート']).fillna(overall_3rd_rate)
df_encoded['trainer_place_turf_dirt_rate'] = df_encoded['trainer_key'].map(te_maps['trainer_key']).fillna(overall_3rd_rate)

feats_for_stacking = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate', 'trainer_place_turf_dirt_rate']

stack_rank_final = lgb.LGBMRanker(n_estimators=300, objective='lambdarank', **lgb_params).fit(
    df_encoded[feats_for_stacking], 19 - df_encoded['着順'].clip(upper=18), group=df_encoded.groupby('レースキー').size().to_numpy()
)
stack_class_final = lgb.LGBMClassifier(n_estimators=300, objective='binary', **lgb_params).fit(
    df_encoded[feats_for_stacking], (df_encoded['着順'] <= 3).astype(int)
)
stack_reg_final = lgb.LGBMRegressor(n_estimators=300, objective='regression', **lgb_params).fit(
    df_encoded[feats_for_stacking], df_encoded['着順']
)
joblib.dump(stack_rank_final, os.path.join(MODEL_DIR, 'stack_rank.pkl'))
joblib.dump(stack_class_final, os.path.join(MODEL_DIR, 'stack_class.pkl'))
joblib.dump(stack_reg_final, os.path.join(MODEL_DIR, 'stack_reg.pkl'))
print("-> [保存成功] stack_rank.pkl / stack_class.pkl / stack_reg.pkl")

# ④ メタモデル予測スコア（Stacking_Score）の全体付与と、最終個別ブレンドモデル（Mシリーズ3種）の全力学習
print("-> 最終個別ブレンド用アセット（M_Rank/M_Class/M_Reg）の全力学習中...")
scaler_global = MinMaxScaler()
r_p_meta = scaler_global.fit_transform(stack_rank_final.predict(df_encoded[feats_for_stacking]).reshape(-1, 1)).flatten()
c_p_meta = stack_class_final.predict_proba(df_encoded[feats_for_stacking])[:, 1]
reg_p_meta = scaler_global.fit_transform((-stack_reg_final.predict(df_encoded[feats_for_stacking])).reshape(-1, 1)).flatten()

meta_df = pd.DataFrame({'r_p': r_p_meta, 'c_p': c_p_meta, 'reg_p': reg_p_meta})
for ccol in meta_context_cols:
    meta_df[ccol] = df_encoded[ccol].values
    
df_encoded['Stacking_Score'] = meta_model.predict_proba(meta_df)[:, 1]
current_feats = feats_for_stacking + ['Stacking_Score']

# 全データに対する最終重みの計算
df_encoded['weight'] = df_encoded.apply(calculate_complex_weight_v3, axis=1)
w_all = df_encoded['weight'].values

m_rank_final = lgb.LGBMRanker(n_estimators=500, objective='lambdarank', **lgb_params).fit(
    df_encoded[current_feats], 19 - df_encoded['着順'].clip(upper=18), group=df_encoded.groupby('レースキー').size().to_numpy(), sample_weight=w_all
)
m_class_final = lgb.LGBMClassifier(n_estimators=500, objective='binary', **lgb_params).fit(
    df_encoded[current_feats], (df_encoded['着順'] <= 3).astype(int), sample_weight=w_all
)
m_reg_final = lgb.LGBMRegressor(n_estimators=500, objective='regression', **lgb_params).fit(
    df_encoded[current_feats], df_encoded['着順'], sample_weight=w_all
)
joblib.dump(m_rank_final, os.path.join(MODEL_DIR, 'm_rank.pkl'))
joblib.dump(m_class_final, os.path.join(MODEL_DIR, 'm_class.pkl'))
joblib.dump(m_reg_final, os.path.join(MODEL_DIR, 'm_reg.pkl'))
print("-> [保存成功] m_rank.pkl / m_class.pkl / m_reg.pkl")

print("\n" + "="*50)
print(" ✨ すべての処理、および『全10場対応・洋芝超偏重メタモデル』と全推論用アセットの完全保存が成功いたしました！")
print("="*50)