import pandas as pd
import numpy as np
import re
import joblib
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# ==========================================
# 既存コードを変更せずに追加した警告・ログ非表示設定
# ==========================================
import warnings
import logging
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# LightGBMのC++側の出力を抑制し、Python側のロガーを無効化（既存処理への影響なし）
lgb_logger = logging.getLogger('lightgbm')
lgb_logger.setLevel(logging.ERROR)
for handler in lgb_logger.handlers[:]:
    lgb_logger.removeHandler(handler)
lgb_logger.addHandler(logging.NullHandler())
lgb_logger.propagate = False

# パラメータのデフォルトにログ出力を抑制する設定を注入
lgb_params_default = {'verbose': -1, 'importance_type': 'split'}
# ==========================================

# ==========================================
# 0. 設定と前処理補助関数
# ==========================================
print("データ読み込み中...")
df = pd.read_csv(r'C:\keiba_AI\final\processed_12_data.csv', low_memory=False)

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
df['馬体重'] = df['馬体重'].fillna(df['馬体重'].mean())
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
    # 基本コンボキー
    t_df['course_id'] = t_df['場所'].astype(str) + '_' + t_df['回り'].astype(str)
    t_df['frame_ratio'] = t_df['馬番'] / t_df['出走数'].replace(0, 1)
    t_df['course_frame_key'] = t_df['course_id'] + '_' + t_df['馬番'].astype(str)
    t_df['course_style_key'] = (
        t_df['場所'].astype(str) + '_' + t_df['回り'].astype(str) + '_' + 
        t_df['距離'].astype(str) + '_' + t_df['脚質ラベル'].astype(str)
    )
    
    # 体重・斤量・能力計算
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
    
    # レース内相対変数の作成
    cols_to_rel = ['weight_ratio', '斤量', '過去平均着順', 'popularity_vs_ability', '年齢', 'full_weight']
    for col in cols_to_rel:
        if col in t_df.columns:
            t_df[f'{col}_rel'] = t_df[col] - t_df.groupby('レースキー')[col].transform('mean')

    # 脚質分布の計算
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
    
    # 複合クロス特徴量群
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
    
    # 性別フラグとクロスエンコーディング
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
print(df.head())

# ==========================================
# 2. 重み付けルール定義
# ==========================================
print("函館特化の重み付け処理中...")
max_year = df['年'].max()

def calculate_complex_weight(row):
    weight = 1.0
    place = str(row['場所'])
    
    if '函館' in place:
        weight *= 5.0
    elif '札幌' in place:
        weight *= 1.5
        
    if row['年'] >= (max_year - 2):
        weight *= 2.0
        
    rank = row['クラス_ランク']
    if rank == 1:
        weight *= 0.7
    elif rank >= 5:
        weight *= 1.3
    
    if '稍' in place: weight *= 1.2
    elif '重' in place: weight *= 1.4
    elif '不' in place: weight *= 1.6

    if row.get('Stacking_Score', 0) > 0.8:
        weight *= 1.5

    return weight

df['weight'] = df.apply(calculate_complex_weight, axis=1)

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
joblib.dump(encoders, 'label_encoders.pkl')
print("エンコーダーを保存しました。")

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
# 5. ターゲットエンコーディング用辞書の生成
# ==========================================
overall_3rd_rate = (df['着順'] <= 3).mean()

def calculate_smooth_rate(group):
    return (group.sum() + 10 * overall_3rd_rate) / (len(group) + 10)

print("統計辞書の作成中...")
stats_dict = {
    'course_frame_rate': df.groupby('course_frame_key')['着順'].apply(lambda x: calculate_smooth_rate((x <= 3).astype(int))).to_dict(),
    'course_style_rate': df.groupby('course_style_key')['着順'].apply(lambda x: calculate_smooth_rate((x <= 3).astype(int))).to_dict(),
    'jockey_place_turf_dirt_rate': df.groupby('騎手_競馬場_芝ダート')['着順'].apply(lambda x: calculate_smooth_rate((x <= 3).astype(int))).to_dict(),
    'trainer_place_turf_dirt_rate': df.groupby(['調教師名', '場所', '芝ダート'])['着順'].apply(lambda x: calculate_smooth_rate((x <= 3).astype(int))).to_dict(),
    'baseline': overall_3rd_rate
}
joblib.dump(stats_dict, 'stats_dict.pkl')
print("統計辞書を保存しました。")

# ==========================================
# 6. 騎手補正・デバフ用ロジック（PKL対応版）
# ==========================================
def apply_jockey_boost_v2(local_df, jockey_efficiency_dict):
    if '騎手' not in local_df.columns:
        return local_df
        
    efficiencies = list(jockey_efficiency_dict.values())
    x0 = np.mean(efficiencies) if len(efficiencies) > 0 else 1.0
    k = 0.5 
    
    def get_sigmoid_boost(eff):
        sigmoid_val = 1 / (1 + np.exp(-k * (eff - x0)))
        return 1.0 + (sigmoid_val * 0.2)

    boost_map = {jockey: get_sigmoid_boost(eff) for jockey, eff in jockey_efficiency_dict.items()}
    local_df['Model_A'] = local_df['Model_A'] * local_df['騎手'].map(boost_map).fillna(1.0)
    return local_df

def apply_final_correction_v2(local_df, jockey_efficiency_dict):
    if '騎手' not in local_df.columns:
        return local_df

    bad_jockeys = [jockey for jockey, eff in jockey_efficiency_dict.items() if eff < 0.3]
    high_perf_jockeys = [jockey for jockey, eff in jockey_efficiency_dict.items() if eff >= 2.0]
    
    if high_perf_jockeys:
        local_df.loc[local_df['騎手'].isin(high_perf_jockeys), 'Model_A'] *= 1.1
    
    if bad_jockeys:
        print(f"\n[自動デバフ適用] 過去データから低効率と判定された騎手への補正: {bad_jockeys}")
        local_df.loc[local_df['騎手'].isin(bad_jockeys), 'Model_A'] *= 0.85
    
    return local_df

# ==========================================
# 7. 分析・レポーティング関数
# ==========================================
def analyze_universal_conditions(local_df, target_year):
    try:
        encoders = joblib.load('label_encoders.pkl')
        readable_df = local_df.copy()
        for col, le in encoders.items():
            if col in readable_df.columns:
                readable_df[col] = le.inverse_transform(readable_df[col])
    except FileNotFoundError:
        print("警告: label_encoders.pkl が見つかりません。数字のまま分析します。")
        readable_df = local_df.copy()

    if not hasattr(analyze_universal_conditions, "cumulative_missed"):
        analyze_universal_conditions.cumulative_missed = []
        analyze_universal_conditions.cumulative_failed = []
        analyze_universal_conditions.all_history_df = pd.DataFrame()

    median_score = readable_df['Model_A'].median()
    top3_pred = readable_df.sort_values(by='Model_A', ascending=False).groupby('レースID').head(5)
    
    failed = top3_pred[top3_pred['着順'] > 3]
    missed = readable_df[(readable_df['Model_A'] < median_score) & (readable_df['着順'] <= 3)]
    
    analyze_universal_conditions.cumulative_missed.append(missed)
    analyze_universal_conditions.cumulative_failed.append(failed)
    analyze_universal_conditions.all_history_df = pd.concat([analyze_universal_conditions.all_history_df, readable_df])
    
    all_missed = pd.concat(analyze_universal_conditions.cumulative_missed)
    all_failed = pd.concat(analyze_universal_conditions.cumulative_failed)

    if '騎手' in readable_df.columns:
        jockey_stats = readable_df.groupby('騎手').agg({
            '着順': lambda x: (x <= 3).sum(), 
            '単勝': lambda x: (1 / x).sum() 
        })
        jockey_stats['efficiency'] = jockey_stats['着順'] / jockey_stats['単勝']
        print(f"\n=== 騎手の真の実力ランキング (期待値に対する効率) ===")
        print(jockey_stats.sort_values(by='efficiency', ascending=False).head(5))
    
    universal_cols = ['騎手', '距離_脚質', '調教師名', 'jockey_trainer', 'frame_ratio']
    print(f"\n=== 【{target_year}年】 函館開催 統計分析レポート ===")
    
    for col in universal_cols:
        if col in readable_df.columns:
            top_missed_cum = all_missed[col].value_counts().index[:5].tolist()
            top_failed_cum = all_failed[col].value_counts().index[:5].tolist()
            
            ratio_year = (missed[col].value_counts() / readable_df[col].value_counts()).fillna(0).sort_values(ascending=False)
            ratio_cum = (all_missed[col].value_counts() / analyze_universal_conditions.all_history_df[col].value_counts()).fillna(0).sort_values(ascending=False)
            
            print(f"\n--- {col} の分析 ---")
            print(f"【累計：穴馬(モデル評価は低かったが3着に入った穴ウマ)】: {top_missed_cum}")
            print(f"【累計：上位予想したが馬券外のカス】: {top_failed_cum}")
            print(f"【{target_year}年：的中率の高い穴馬の条件 (出現率TOP5)】:")
            for name, val in ratio_year.head(5).items():
                print(f"  {name}: {val:.2%}")
            print(f"【累計：的中率の高い穴馬の条件 (出現率TOP5)】:")
            for name, val in ratio_cum.head(5).items():
                print(f"  {name}: {val:.2%}")
        else:
            print(f"\n※ データに '{col}' が存在しません。")

def evaluate_and_print_results(test_df, target_year):
    test_df = test_df.copy()
    test_df['place_code'] = test_df['レースID'].astype(str).str[4:6]
    models = ['Model_A', 'Model_B', 'Model_C', 'Model_D']
    
    print(f"\n=== {target_year}年 全体評価 ===")
    for m in models:
        print(f"--- {m} ---")
        print(evaluate_detailed(test_df, m))
        
    target_code = '02'
    local_df = test_df[test_df['place_code'] == target_code].copy()
    
    if not local_df.empty:
        print(f"\n=== {target_year}年 函館開催（場所コード:{target_code}）評価 ===")
        for m in models:
            print(f"--- {m} ---")
            print(evaluate_detailed(local_df, m))
        
        print(f"\n=== 穴馬/不適格馬：予測分析レポート ===")
        local_df = local_df.sort_values(by=['レースID', 'Model_A'], ascending=[True, False])
        
        top3_pred = local_df.groupby('レースID').head(3).copy()
        top3_pred['is_correct'] = (top3_pred['着順'] <= 3).astype(int)
        
        failed_list = top3_pred[top3_pred['is_correct'] == 0]
        missed_list = local_df[(local_df['Model_A'] < local_df['Model_A'].median()) & (local_df['着順'] <= 3)]
        
        print(f"\n【上位予想したが3着以内外】\n{failed_list[['レースID', '馬名', 'Model_A', '着順']].head(5)}")
        print(f"\n【見落とした穴馬（スコア低めだが3着以内）】\n{missed_list[['レースID', '馬名', 'Model_A', '着順']].head(5)}")
        
        analyze_universal_conditions(local_df, target_year)
        print(local_df)
    else:
        print(f"\n函館開催のデータは見つかりませんでした。")

# ==========================================
# 8. バックテスト＆モデリングの実行（時系列ローリング・スタッキング完全対応）
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

all_results = []
# バックテスト（本番）は 2016 年からスタート
years = [y for y in sorted(df['年'].unique()) if y >= 2016]

# すべてのデータで一括初期化
df['Stacking_Score'] = 0.0

# 精度向上のためのLightGBMコアパラメータ設定
lgb_params = {
    'learning_rate': 0.03,
    'num_leaves': 31,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': 42,
    'n_jobs': -1,
    **lgb_params_default  # 警告抑制パラメータの展開
}

# ------------------------------------------
# 【仕込みフェーズ】2010〜2015年の6年分で最初のメタモデルの土台を作る
# ------------------------------------------
print("\n--- 【仕込み期間】2010〜2015年のベースモデル構築中 ---")
init_train = df[(df['年'] >= 2010) & (df['年'] <= 2015)].copy()

# 仕込み期間内での簡易Out-of-Fold、または時系列統計量
stats_frame_course = init_train.groupby('course_frame_key')['着順'].apply(lambda x: (x <= 3).mean())
stats_course_style = init_train.groupby('course_style_key')['着順'].apply(lambda x: (x <= 3).mean())
stats_jockey_place_turf_dirt = init_train.groupby('騎手_競馬場_芝ダート')['着順'].apply(lambda x: (x <= 3).mean())
baseline_rate = (init_train['着順'] <= 3).mean()

# メタモデルを育てるための特徴量蓄積リスト
meta_features_pool = []
meta_target_pool = []

# 仕込み期間（2011〜2015）を1年ずつ擬似ローリングして、初期のメタ学習用データを溜める
for stage_y in range(2011, 2016):
    stage_tr = init_train[init_train['年'] < stage_y]
    stage_va = init_train[init_train['年'] == stage_y]
    if stage_tr.empty or stage_va.empty: continue
    
    # 3モデルの簡易学習
    features_tmp = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate']
    tmp_rank = lgb.LGBMRanker(n_estimators=150, objective='lambdarank', **lgb_params).fit(
        stage_tr[feat_base], 19 - stage_tr['着順'].clip(upper=18), group=stage_tr.groupby('レースキー').size().to_numpy()
    )
    tmp_class = lgb.LGBMClassifier(n_estimators=150, objective='binary', **lgb_params).fit(
        stage_tr[feat_base], (stage_tr['着順'] <= 3).astype(int)
    )
    tmp_reg = lgb.LGBMRegressor(n_estimators=150, objective='regression', **lgb_params).fit(
        stage_tr[feat_base], stage_tr['着順']
    )
    
    scaler_tmp = MinMaxScaler()
    r_p = scaler_tmp.fit_transform(tmp_rank.predict(stage_va[feat_base]).reshape(-1, 1)).flatten()
    c_p = tmp_class.predict_proba(stage_va[feat_base])[:, 1]
    reg_p = scaler_tmp.fit_transform((-tmp_reg.predict(stage_va[feat_base])).reshape(-1, 1)).flatten()
    
    meta_features_pool.append(pd.DataFrame({'r_p': r_p, 'c_p': c_p, 'reg_p': reg_p}))
    meta_target_pool.append((stage_va['着順'] <= 3).astype(int))

# 初期メタモデルの学習
X_meta_init = pd.concat(meta_features_pool, ignore_index=True)
y_meta_init = pd.concat(meta_target_pool, ignore_index=True)
meta_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, random_state=42, objective='binary', **lgb_params_default)
meta_model.fit(X_meta_init, y_meta_init)


# ------------------------------------------
# 【本番バックテスト期間】1年ずつのローリングループ（2016年〜最新年）
# ------------------------------------------
print("\n--- 【本番バックテスト】時系列ローリングループ開始 ---")
for target_year in tqdm(years, desc="本番ローリング実行中"):
    # 1. 【学習データの確定】ターゲット前年までの全履歴
    train = df[(df['年'] >= 2010) & (df['年'] < target_year)].copy()
    test = df[df['年'] == target_year].copy()
    
    if train.empty or test.empty: continue

    # 2. 【騎手・コース統計の算出】（未来リークの完全排除）
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
    joblib.dump(jockey_eff_dict, 'jockey_efficiency_backtest.pkl')

    stats_frame_course = train.groupby('course_frame_key')['着順'].apply(lambda x: (x <= 3).mean())
    stats_course_style = train.groupby('course_style_key')['着順'].apply(lambda x: (x <= 3).mean())
    stats_jockey_place_turf_dirt = train.groupby('騎手_競馬場_芝ダート')['着順'].apply(lambda x: (x <= 3).mean())
    stats_trainer_place_turf_dirt = train.groupby(['調教師名', '場所', '芝ダート'])['着順'].apply(lambda x: (x <= 3).mean())
    baseline_rate = (train['着順'] <= 3).mean()

    for t_df in [train, test]:
        t_df['course_frame_rate'] = t_df['course_frame_key'].map(stats_frame_course).fillna(baseline_rate)
        t_df['course_style_rate'] = t_df['course_style_key'].map(stats_course_style).fillna(baseline_rate)
        t_df['jockey_place_turf_dirt_rate'] = t_df['騎手_競馬場_芝ダート'].map(stats_jockey_place_turf_dirt).fillna(baseline_rate)
        t_df['trainer_key'] = list(zip(t_df['調教師名'], t_df['場所'], t_df['芝ダート']))
        t_df['trainer_place_turf_dirt_rate'] = t_df['trainer_key'].map(stats_trainer_place_turf_dirt).fillna(baseline_rate)

    # 3. 【最重要：テスト対象年の Stacking_Score を予測・上書き】
    # この時点の「過去データ（train）」からベース3モデルを組み上げて、テスト年のスコアを出す
    feats_for_stacking = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate', 'trainer_place_turf_dirt_rate']
    
    stack_rank = lgb.LGBMRanker(n_estimators=300, objective='lambdarank', **lgb_params).fit(
        train[feats_for_stacking], 19 - train['着順'].clip(upper=18), group=train.groupby('レースキー').size().to_numpy()
    )
    stack_class = lgb.LGBMClassifier(n_estimators=300, objective='binary', **lgb_params).fit(
        train[feats_for_stacking], (train['着順'] <= 3).astype(int)
    )
    stack_reg = lgb.LGBMRegressor(n_estimators=300, objective='regression', **lgb_params).fit(
        train[feats_for_stacking], train['着順']
    )
    
    scaler_test = MinMaxScaler()
    test_r_p = scaler_test.fit_transform(stack_rank.predict(test[feats_for_stacking]).reshape(-1, 1)).flatten()
    test_c_p = stack_class.predict_proba(test[feats_for_stacking])[:, 1]
    test_reg_p = scaler_test.fit_transform((-stack_reg.predict(test[feats_for_stacking])).reshape(-1, 1)).flatten()
    
    test_batch_df = pd.DataFrame({'r_p': test_r_p, 'c_p': test_c_p, 'reg_p': test_reg_p})
    
    # 直近までのメタモデルで、今年の純粋な予測スコアを出して、オリジナルDataFrameに書き戻す
    test['Stacking_Score'] = meta_model.predict_proba(test_batch_df)[:, 1]
    df.loc[test.index, 'Stacking_Score'] = test['Stacking_Score']

    # 過去データ側も同様のベース予測を現在のtrainに反映
    train_r_p = scaler_test.fit_transform(stack_rank.predict(train[feats_for_stacking]).reshape(-1, 1)).flatten()
    train_c_p = stack_class.predict_proba(train[feats_for_stacking])[:, 1]
    train_reg_p = scaler_test.fit_transform((-stack_reg.predict(train[feats_for_stacking])).reshape(-1, 1)).flatten()
    train_batch_df = pd.DataFrame({'r_p': train_r_p, 'c_p': train_c_p, 'reg_p': train_reg_p})
    train['Stacking_Score'] = meta_model.predict_proba(train_batch_df)[:, 1]

    # 4. 【本番メインモデル学習】クリーンなスコアが入った特徴量で最終予測へ
    current_feats = feats_for_stacking + ['Stacking_Score']
    train_weights = train.apply(calculate_complex_weight, axis=1).values
    
    # バリデーション分割（アーリーストッピング用）
    val_size = int(len(train) * 0.15)
    train_idx, val_idx = train.index[:-val_size], train.index[-val_size:]
    
    X_tr, y_tr = train.loc[train_idx, current_feats], train.loc[train_idx]
    X_val, y_val = train.loc[val_idx, current_feats], train.loc[val_idx]
    w_tr, w_val = train_weights[:-val_size], train_weights[-val_size:]

    # メインモデルの学習実行
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

    # 5. 【テストデータへの予測とブレンド】
    r_p_final = scaler_test.fit_transform(m_rank.predict(test[current_feats]).reshape(-1, 1)).flatten()
    c_p_final = m_class.predict_proba(test[current_feats])[:, 1]
    reg_p_final = scaler_test.fit_transform((-m_reg.predict(test[current_feats])).reshape(-1, 1)).flatten()
    
    test['Model_A'] = (r_p_final * 0.4) + (c_p_final * 0.3) + (reg_p_final * 0.3)
    test['Model_B'] = (r_p_final * 0.2) + (c_p_final * 0.6) + (reg_p_final * 0.2)
    test['Model_C'] = (c_p_final * 0.5) + (reg_p_final * 0.5)
    test['Model_D'] = (c_p_final)
    
    # 騎手補正と最終デバフ
    loaded_jockey_eff = joblib.load('jockey_efficiency_backtest.pkl')
    test = apply_jockey_boost_v2(test, loaded_jockey_eff)
    test = apply_final_correction_v2(test, loaded_jockey_eff)

    # 評価結果の表示
    evaluate_and_print_results(test, target_year)
    all_results.append(test)
    
    # 6. 【メタモデルの更新】今年のテスト結果（予測値と正解ラベル）を履歴に追加し、来年のためにメタモデルを再学習
    meta_features_pool.append(test_batch_df)
    meta_target_pool.append((test['着順'] <= 3).astype(int))
    
    X_meta_updated = pd.concat(meta_features_pool, ignore_index=True)
    y_meta_updated = pd.concat(meta_target_pool, ignore_index=True)
    meta_model.fit(X_meta_updated, y_meta_updated)

final = pd.concat(all_results)

# ==========================================
# 9. メタモデルの最終決定と可視化
# ==========================================
print("メタモデル（スタッキング）の学習を開始...")
X_meta = pd.concat(meta_features_pool, ignore_index=True)
y_meta = pd.concat(meta_target_pool, ignore_index=True)

meta_model = lgb.LGBMClassifier(
    n_estimators=300, 
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=30,
    random_state=42,
    objective='binary',
    **lgb_params_default
)
meta_model.fit(X_meta, y_meta)
print("メタモデルの学習が完了しました。")

meta_probs = meta_model.predict_proba(X_meta)[:, 1]
final['Stacking_Score'] = meta_probs

print("\n=== メタモデルの判断基準（重要度） ===")
meta_importance = pd.DataFrame({'feature': ['r_p', 'c_p', 'reg_p'], 'importance': meta_model.feature_importances_}).sort_values(by='importance', ascending=False)
print(meta_importance)

print("\n=== メタモデルの予測スコアの分布（期待値など） ===")
print(pd.Series(meta_probs).describe())

# ==========================================
# 10. 全データを用いた本番用最終学習 【★学習ロジック最適化箇所】
# ==========================================
print("--- 全期間データでの最終モデル学習を開始 ---")
final_groups = df.groupby('レースキー').size().to_numpy()
df['trainer_key'] = list(zip(df['調教師名'], df['場所'], df['芝ダート']))

df['course_frame_rate'] = df['course_frame_key'].map(df.groupby('course_frame_key')['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)
df['course_style_rate'] = df['course_style_key'].map(df.groupby('course_style_key')['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)
df['jockey_place_turf_dirt_rate'] = df['騎手_競馬場_芝ダート'].map(df.groupby('騎手_競馬場_芝ダート')['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)
df['trainer_place_turf_dirt_rate'] = df['trainer_key'].map(df.groupby(['調教師名', '場所', '芝ダート'])['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)

print("全データ用スタッキング特徴量の算出中...")
final_feats_without_stacking = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate', 'trainer_place_turf_dirt_rate']

# 本番用はデータ量が最大化されているため、バックテスト時よりやや深く学習(n_estimatorsを調整)
tmp_rank = lgb.LGBMRanker(n_estimators=450, objective='lambdarank', **lgb_params).fit(df[final_feats_without_stacking], 19 - df['着順'].clip(upper=18), group=final_groups)
tmp_class = lgb.LGBMClassifier(n_estimators=450, objective='binary', **lgb_params).fit(df[final_feats_without_stacking], (df['着順'] <= 3).astype(int))
tmp_reg = lgb.LGBMRegressor(n_estimators=450, objective='regression', **lgb_params).fit(df[final_feats_without_stacking], df['着順'])

scaler_tmp = MinMaxScaler()
all_r_p = scaler_tmp.fit_transform(tmp_rank.predict(df[final_feats_without_stacking]).reshape(-1, 1)).flatten()
all_c_p = tmp_class.predict_proba(df[final_feats_without_stacking])[:, 1]
all_reg_p = scaler_tmp.fit_transform((-tmp_reg.predict(df[final_feats_without_stacking])).reshape(-1, 1)).flatten()

all_batch_df = pd.DataFrame({'r_p': all_r_p, 'c_p': all_c_p, 'reg_p': all_reg_p})
df['Stacking_Score'] = meta_model.predict_proba(all_batch_df)[:, 1]

final_feats = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate', 'trainer_place_turf_dirt_rate', 'Stacking_Score']
print(final_feats)
print("最終モデルの学習中...")

# 最終保存用モデルのフィッティング
final_m_rank = lgb.LGBMRanker(n_estimators=450, objective='lambdarank', **lgb_params).fit(df[final_feats], 19 - df['着順'].clip(upper=18), group=final_groups)
final_m_class = lgb.LGBMClassifier(n_estimators=450, objective='binary', **lgb_params).fit(df[final_feats], (df['着順'] <= 3).astype(int))
final_m_reg = lgb.LGBMRegressor(n_estimators=450, objective='regression', **lgb_params).fit(df[final_feats], df['着順'])

# ------------------------------------------
# 本番推論用（全期間データ）の騎手効率辞書をPKL保存
# ------------------------------------------
print("本番推論用の最終騎手効率辞書を保存中...")
final_jockey_counts = df['騎手'].value_counts()
final_valid_jockeys = final_jockey_counts[final_jockey_counts >= 5].index

if len(final_valid_jockeys) > 0:
    final_jockey_stats = df[df['騎手'].isin(final_valid_jockeys)].groupby('騎手').agg({
        '着順': lambda x: (x <= 3).sum(),
        '単勝': lambda x: (1 / x).replace(np.inf, 0).sum()
    })
    final_jockey_eff_dict = (final_jockey_stats['着順'] / final_jockey_stats['単勝'].replace(0, 1)).to_dict()
else:
    final_jockey_eff_dict = {}

joblib.dump(final_jockey_eff_dict, 'jockey_efficiency_final.pkl')

# ==========================================
# 11. 各種オブジェクト・スケーラーの保存
# ==========================================
joblib.dump(final_m_rank, 'model_rank.pkl')
joblib.dump(final_m_class, 'model_class.pkl')
joblib.dump(final_m_reg, 'model_reg.pkl')
joblib.dump(final_feats, 'feature_names.pkl') 
joblib.dump(meta_model, 'meta_model.pkl')

print("推論用の正規化スケーラーを保存中...")
r_p_all = final_m_rank.predict(df[final_feats]).reshape(-1, 1)
c_p_all = final_m_class.predict_proba(df[final_feats])[:, 1].reshape(-1, 1)
reg_p_all = (-final_m_reg.predict(df[final_feats])).reshape(-1, 1)