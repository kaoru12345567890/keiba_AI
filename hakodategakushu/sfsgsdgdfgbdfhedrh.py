# ==========================================
#   ＜＜データの特徴量選定と前処理の方針＞＞
#-----------------------------------------
#    ＜ベースデータにあり、使える初期要素＞
# レースID,年,月,日,場所,回,日目,レース目,
# レース名,天気,馬場状態,レース条件,芝ダート,距離,
# 回り,出走数,着順,枠番,馬番,馬名,性別年齢,斤量,騎手,
# タイム,着差,ペース,通過順,上り3ハロン,単勝,人気,馬体重,体重増減,
# 所属,調教師名,馬主,賞金,脚質スコア,脚質ラベル,上がり偏差値,性別,年齢,
# 過去出走回数,過去平均着順,過去連対率,過去複勝率,過去平均上がり偏差値,
#   ->増えたよまあ全部使おうかな
#  
#-----------------------------------------
#    ＜リークの可能性がある要素＞
# (着順),上り3ハロン,単勝,人気,上がり偏差値,タイム,ペース,
#-----------------------------------------
#    ＜今回のモデルで全く使わない要素＞
# 曜日,着差,通過順,上り3ハロン,上がり偏差値,人気,馬主,賞金,脚質スコア,性別年齢,タイム,ペース,
#-----------------------------------------
#    ＜今回のモデルで使う要素＞
# 年,月,日,場所,回,日目,レース目,天気,馬場状態,芝ダート,
# 回り,距離,出走数,枠番,馬番,斤量,(着順),馬体重,体重増減,
# 性別,年齢,騎手,所属,調教師名,脚質ラベル,単勝,
# 過去出走回数,過去平均着順,過去連対率,過去複勝率,過去平均上がり偏差値
#-----------------------------------------
#     ＜使いたけど使い方がわからない要素＞
# レースID,レース名,馬名,
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import re
import joblib
from sklearn.model_selection import KFold

# 1. 初期ファイル読み込み
print("データ読み込み中...")
df = pd.read_csv(r'C:\keiba_AI\final\processed_12_data.csv', low_memory=False)

# 数字化処理（floatに変換）
print("数値化処理中...")
numeric_cols = ['年', '月', '日', 'レース目', '出走数', '枠番', '馬番',
                '斤量', '着順', '距離', '単勝','過去平均着順', '過去出走回数',
                '過去連対率', '過去複勝率', '過去平均上がり偏差値', '年齢', '馬体重']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

# 体重増減の処理を追加
def clean_weight_diff(x):
    x_str = str(x).strip()
    if '計不' in x_str or x_str == 'nan' or x_str == '':
        return np.nan 
    match = re.search(r'([+-]?\d+)', x_str)
    if match:
        return float(match.group(0))
    return 0.0

    
print("馬体重の処理中...")
df['体重増減'] = df['体重増減'].apply(clean_weight_diff).fillna(0.0)
df['馬体重'] = df['馬体重'].fillna(df['馬体重'].mean())
df['過去平均着順'] = df['過去平均着順'].fillna(7.5)

# ========================================================
# 1. レース名から「クラス_ランク」と「新馬戦フラグ」を作る関数
# ========================================================
print("クラス_ランクと新馬戦フラグの作成中...")
df['is_新馬戦'] = df['レース名'].str.contains('新馬', na=False).astype(int)

# applyによる条件分岐を np.select に置き換え（大幅な高速化）
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
        t_df['場所'].astype(str) + '_' + 
        t_df['回り'].astype(str) + '_' + 
        t_df['距離'].astype(str) + '_' + 
        t_df['脚質ラベル'].astype(str)
    )
    
    t_df['full_weight'] = t_df['馬体重'] + t_df['斤量']
    t_df['weight_ratio'] = t_df['馬体重'] / t_df['斤量'].replace(0, 1)
    t_df['weight_diff'] = t_df['馬体重'] - t_df['斤量'] * 8
    t_df['popularity_vs_ability'] = (np.log1p(t_df['単勝']) / (t_df['過去平均着順'] + 1))
    t_df['weight_age_ratio'] = t_df['馬体重'] * t_df['年齢']
    t_df['jockey_trainer'] = t_df['騎手'].astype(str) + '_' + t_df['調教師名'].astype(str)
    t_df['レースキー'] = t_df['年'].astype(int).astype(str) + t_df['月'].astype(int).astype(str) + t_df['日'].astype(int).astype(str) + t_df['場所'].astype(str) + t_df['レース目'].astype(int).astype(str)
    
    # グループ変換（平均差分）
    cols_to_rel = ['weight_ratio', '斤量', '過去平均着順', 'popularity_vs_ability', '年齢', 'full_weight']
    for col in cols_to_rel:
        if col in t_df.columns:
            t_df[f'{col}_rel'] = t_df[col] - t_df.groupby('レースキー')[col].transform('mean')

    # 脚質関連のループ処理を最適化
    for i in [1, 2, 3, 4]:
        col_name_i = f'is_temp_脚質{i}'
        t_df[col_name_i] = (t_df['脚質ラベル'] == i).astype(int)
        
        t_df[f'脚質{i}_頭数'] = t_df.groupby('レースキー')[col_name_i].transform('sum')
        t_df[f'脚質{i}_割合'] = t_df[f'脚質{i}_頭数'] / t_df['出走数'].replace(0, 1)
        
        t_df.drop(columns=[col_name_i], inplace=True)
    
    # 同型ライバル頭数（高速化）
    t_df['同型ライバル頭数'] = 0
    for i in [1, 2, 3, 4]:
        mask = (t_df['脚質ラベル'] == i)
        t_df.loc[mask, '同型ライバル頭数'] = t_df.loc[mask, f'脚質{i}_頭数'] - 1
    
    # <新しく追加した特徴量のリスト>
    t_df['騎手_競馬場_芝ダート'] = t_df['騎手'].astype(str) + '_' + t_df['場所'] + '_' + t_df['芝ダート']
    t_df['jockey_脚質'] = t_df['騎手'].astype(str) + '_' + t_df['脚質ラベル'].astype(str)
    t_df['場所_脚質'] = t_df['場所'].astype(str) + '_' + t_df['脚質ラベル'].astype(str)+ '_' + t_df['芝ダート']
    t_df['馬場状態_脚質'] = t_df['馬場状態'].astype(str) + '_' + t_df['脚質ラベル'].astype(str)
    t_df['場所_芝ダート_馬場状態'] = t_df['場所'].astype(str) + '_' + t_df['芝ダート'].astype(str) + '_' + t_df['馬場状態'].astype(str)
    t_df['天気_脚質'] = t_df['天気'].astype(str) + '_' + t_df['脚質ラベル'].astype(str)
    t_df['天気_騎手'] = t_df['天気'].astype(str) + '_' + t_df['騎手'].astype(str)
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
    t_df['距離_脚質_馬場状態'] = t_df['距離_脚質'].astype(str)+ '_' + t_df['馬場状態'].astype(str)
    t_df['is_senba'] = (t_df['性別'] == 1).astype(int)  # セン馬
    t_df['is_female'] = (t_df['性別'] == 2).astype(int) # 牝馬
    t_df['is_male'] = (t_df['性別'] == 3).astype(int)   # 牡馬

    # カテゴリの結合（年齢や斤量relは文字列化して結合）
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
# 函館特化・直近・レースランクの重み付け処理
# ==========================================
print("函館特化の重み付け処理中...")
max_year = df['年'].max()

def calculate_complex_weight(row):
    weight = 1.0
    
    # ① 場所の重み（絶対条件）
    place = str(row['場所'])
    if '函館' in place:
        weight *= 5.0  # 函館のレースは重視
    elif '札幌' in place:
        weight *= 1.5  # 札幌のレースは中程度に重視
        
    # ② 直近3年の重み（最新トレンドの重視）
    if row['年'] >= (max_year - 2):  # 当年、前年、前々年
        weight *= 2.0
        
    # ③ クラスの重み（ノイズの多い下級条件を割引）
    rank = row['クラス_ランク']
    if rank == 1:
        weight *= 0.7  # 新馬・未勝利はノイズが多いので30%オフ
    elif rank >= 5:
        weight *= 1.3  # OP・重賞は実力通りので20%マシ
    
    field = str(row['場所'])
    if '稍' in field:
        weight *= 1.2  # 馬場状態が稍重の場合は10%増し
    elif '重' in field:
        weight *= 1.4  # 馬場状態が重の場合は20%増し
    elif '不' in field:
        weight *= 1.6  # 馬場状態が不良の場合は30%増し

    if row.get('Stacking_Score', 0) > 0.8:
        weight *= 1.5

    return weight

df['weight'] = df.apply(calculate_complex_weight, axis=1)

# ==========================================
# 2. カテゴリ変数のLabel Encodingと保存
# ==========================================
print("カテゴリ変数のエンコーディング中...")
categorical_cols = [
    '場所', '回り', "芝ダート", '天気', '馬場状態', '騎手', '所属', 
    '調教師名', 'course_id', 'course_frame_key', 
    'course_style_key', 'jockey_trainer',
    '騎手_競馬場_芝ダート', 
    'jockey_脚質', '場所_脚質', '馬場状態_脚質', '場所_芝ダート_馬場状態','天気_脚質', '天気_騎手',
    '調教師名_所属','調教師名_is_新馬戦','騎手_is_新馬戦','調教師名_is_ハンデ戦','騎手_is_ハンデ戦',
    'hakodate_jockey_trainer', '調教師名_is_牝馬限定','騎手_is_牝馬限定','調教師名_斤量_ルール',
    '騎手_斤量_ルール','距離_脚質_馬場状態','距離_脚質','騙馬年齢','牝馬年齢','牡馬年齢',
    '騙馬斤量_rel','牝馬斤量_rel','牡馬斤量_rel',
    '1走前_芝ダート','2走前_芝ダート','3走前_芝ダート',
    '1走前_着順','2走前_着順','3走前_着順'
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
# 評価関数の定義（詳細版）
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
# 3. 本番用統計辞書の作成と保存
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
# 4. 新規：リークを防ぐ過去データベースの騎手補正関数
# ==========================================
def apply_jockey_boost_v2(local_df, jockey_efficiency_dict):
    """ 前年までに蓄積された騎手効率辞書に基づき、シグモイド関数で滑らかな補正倍率を適用 """
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
    """ 前年までに蓄積された騎手効率辞書に基づき、自動でデバフ・バフをかける """
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
# 函館専用関数（元の評価と分析レポート出力用）
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
    models = ['Model_A', 'Model_B', 'Model_C', 'Model_D' ]
    
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
        
        print(f"\n=== 函館開催：予測分析レポート ===")
        local_df = local_df.sort_values(by=['レースID', 'Model_A'], ascending=[True, False])
        
        top3_pred = local_df.groupby('レースID').head(3).copy()
        top3_pred['is_correct'] = (top3_pred['着順'] <= 3).astype(int)
        
        failed_list = top3_pred[top3_pred['is_correct'] == 0]
        missed_list = local_df[(local_df['Model_A'] < local_df['Model_A'].median()) & (local_df['着順'] <= 3)]
        
        print(f"\n【上位予想だが3着以内外】\n{failed_list[['レースID', '馬名', 'Model_A', '着順']].head(5)}")
        print(f"\n【見落とした穴馬（スコア低めだが3着以内）】\n{missed_list[['レースID', '馬名', 'Model_A', '着順']].head(5)}")
        
        analyze_universal_conditions(local_df, target_year)
        print(local_df)
    else:
        print(f"\n函館開催のデータは見つかりませんでした。")

# ==========================================
# 5. 学習の実行（年度別の順次バックテスト）
# ==========================================
print("モデルの学習開始...")

drop_cols = [
    '着順', 
    'レースID', 'レースキー', 'レース名', '馬名','popularity_vs_ability',
    '年', '月', '日', '曜日', 'レース条件', '脚質ラベル','ペース',
    'weight', "性別年齢", "タイム", "着差", "通過順", "馬主", "賞金",
    '回', '日目', 'レース目', '斤量', '上り3ハロン', '単勝', '人気', 'full_weight',
    '馬体重', '体重増減', '脚質スコア', '上がり偏差値','天気','場所','斤量_ルール','騎手',
    'course_id','馬場状態','回り','枠番','年齢', '馬番', 'is_ハンデ戦','性別',
    'is_牝馬限定', 'is_新馬戦','所属','脚質1_頭数','脚質2_頭数','脚質3_頭数',
    '脚質4_頭数','芝ダート','is_hakodate','距離','is_senba','is_female','is_male',
    '前走との間隔_週数', '3走前_芝ダート', '2走前_芝ダート', '牡馬斤量_rel', '牝馬斤量_rel'
]
feats = [c for c in df.columns if c not in drop_cols]
print(feats)

train_X = df[feats]
train_y = df['着順']
train_w = df['weight']
groups = df.groupby('レースキー').size().to_numpy()

feat_base = [c for c in df.columns if c not in drop_cols]

all_results = []
print("--- 統合適性スコアを含む学習パイプライン実行中 ---")
years = [y for y in sorted(df['年'].unique()) if y >= 2021]

meta_train_features = [] 
meta_train_target = []

meta_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
meta_pkl_path = 'meta_model.pkl'

if 'Stacking_Score' not in df.columns:
    df['Stacking_Score'] = 0.0

# 後述する全期間メタ予測のために、各年度の訓練済みベースモデル群を一時保管するリスト
trained_base_models = []

for target_year in tqdm(years, desc="年度別学習"):
    train = df[(df['年'] >= 2010) & (df['年'] < target_year)].copy()
    test = df[df['年'] == target_year].copy()
    
    if train.empty or test.empty: continue

    df.loc[test.index, 'Stacking_Score'] = test['Stacking_Score']

    # ==========================================
    # 【新規】前年までの学習データから騎手効率を計算・保存 (最初はデータ0でも安全に処理)
    # ==========================================
    jockey_counts = train['騎手'].value_counts()
    valid_jockeys = jockey_counts[jockey_counts >= 5].index
    
    if len(valid_jockeys) > 0:
        jockey_stats = train[train['騎手'].isin(valid_jockeys)].groupby('騎手').agg({
            '着順': lambda x: (x <= 3).sum(),
            '単勝': lambda x: (1 / x).sum()
        })
        jockey_stats['efficiency'] = jockey_stats['着順'] / jockey_stats['単勝'].replace(0, 1)
        jockey_eff_dict = jockey_stats['efficiency'].to_dict()
    else:
        jockey_eff_dict = {} # 初期データなしの場合は空辞書
        
    joblib.dump(jockey_eff_dict, 'jockey_efficiency.pkl')

    # 統計量の計算
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

    current_feats = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate', 'trainer_place_turf_dirt_rate', 'Stacking_Score']

    train_weights = train.apply(calculate_complex_weight, axis=1).values
    groups_train = train.groupby('レースキー').size().to_numpy()
    
    # LGBMRanker
    m_rank = lgb.LGBMRanker(n_estimators=300).fit(
        train[current_feats], 
        19 - train['着順'].clip(upper=18), 
        group=groups_train,
        sample_weight=train_weights
    )
    
    # LGBMClassifier
    m_class = lgb.LGBMClassifier(n_estimators=300).fit(
        train[current_feats], 
        (train['着順'] <= 3).astype(int),
        sample_weight=train_weights
    )
    
    # LGBMRegressor
    m_reg = lgb.LGBMRegressor(n_estimators=300).fit(
        train[current_feats], 
        train['着順'],
        sample_weight=train_weights
    )
    
    importance_df = pd.DataFrame({
        'feature': current_feats,
        'importance': m_rank.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("\n=== 重要度トップ20（リーク確認用） ===")
    print(importance_df.head(20))
    print("\n=== 重要度下位20（断捨離候補） ===")
    print(importance_df.tail(20).to_string())

    scaler = MinMaxScaler()
    r_p = scaler.fit_transform(m_rank.predict(test[current_feats]).reshape(-1, 1)).flatten()
    c_p = m_class.predict_proba(test[current_feats])[:, 1]
    reg_p = scaler.fit_transform((-m_reg.predict(test[current_feats])).reshape(-1, 1)).flatten()
    
    batch_df = pd.DataFrame({'r_p': r_p, 'c_p': c_p, 'reg_p': reg_p})
    meta_train_features.append(batch_df)
    meta_train_target.append((test['着順'] <= 3).astype(int))

    X_meta = pd.concat(meta_train_features, ignore_index=True)
    y_meta = pd.concat(meta_train_target, ignore_index=True)
    
    if years.index(target_year) > 0: 
        meta_model.fit(X_meta, y_meta)
        joblib.dump(meta_model, meta_pkl_path)

        test['Stacking_Score'] = meta_model.predict_proba(batch_df)[:, 1]
        df.loc[test.index, 'Stacking_Score'] = test['Stacking_Score']

    test['Model_A'] = (r_p * 0.4) + (c_p * 0.3) + (reg_p * 0.3)
    test['Model_B'] = (r_p * 0.2) + (c_p * 0.6) + (reg_p * 0.2)
    test['Model_C'] = (c_p * 0.5) + (reg_p * 0.5)
    test['Model_D'] = (c_p)
    
    # ==========================================
    # 【新規】保存した前年までの騎手効率PKLを呼び出してテストデータをリークなしで補正
    # ==========================================
    try:
        loaded_jockey_eff = joblib.load('jockey_efficiency.pkl')
    except FileNotFoundError:
        loaded_jockey_eff = {}
        
    test = apply_jockey_boost_v2(test, loaded_jockey_eff)
    test = apply_final_correction_v2(test, loaded_jockey_eff)

    evaluate_and_print_results(test, target_year)
    all_results.append(test)

# 全結果を統合
final = pd.concat(all_results)

# ==========================================
# メタモデル（スタッキング）の学習
# ==========================================
print("メタモデル（スタッキング）の学習を開始...")
X_meta = pd.concat(meta_train_features, ignore_index=True)
y_meta = pd.concat(meta_train_target, ignore_index=True)

meta_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
meta_model.fit(X_meta, y_meta)
print("メタモデルの学習が完了しました。")

# ==========================================
# 【修正箇所】予測確率の格納先をdfからfinalへ変更しエラー回避
# ==========================================
meta_probs = meta_model.predict_proba(X_meta)[:, 1]
final['Stacking_Score'] = meta_probs

# ==========================================
# メタモデルの学習結果（重要度）の可視化
# ==========================================
print("\n=== メタモデルの判断基準（重要度） ===")
meta_importance = pd.DataFrame({
    'feature': ['r_p', 'c_p', 'reg_p'],
    'importance': meta_model.feature_importances_
}).sort_values(by='importance', ascending=False)
print(meta_importance)

print("\n=== メタモデルの予測スコアの分布（期待値など） ===")
print(pd.Series(meta_probs).describe())
print("全期間のデータにスタッキングスコアを付与しました。")

# ==========================================
# 1. 全データで最終学習を行う
# ==========================================
print("--- 全期間データでの最終モデル学習を開始 ---")
final_groups = df.groupby('レースキー').size().to_numpy()
df['trainer_key'] = list(zip(df['調教師名'], df['場所'], df['芝ダート']))

df['course_frame_rate'] = df['course_frame_key'].map(df.groupby('course_frame_key')['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)
df['course_style_rate'] = df['course_style_key'].map(df.groupby('course_style_key')['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)
df['jockey_place_turf_dirt_rate'] = df['騎手_競馬場_芝ダート'].map(df.groupby('騎手_競馬場_芝ダート')['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)
df['trainer_place_turf_dirt_rate'] = df['trainer_key'].map(df.groupby(['調教師名', '場所', '芝ダート'])['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)

# 【修正箇所】全期間学習の前に、ベースモデル予測値を一度df全体に対してまとめて作り、Stacking_Scoreのエラーを根本解決します
print("全データ用スタッキング特徴量の算出中...")
final_feats_without_stacking = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate', 'trainer_place_turf_dirt_rate']

# 全期間に対してベースモデルの予測スコアを作るために、一度現在の状態の仮ベースモデルをフィット
tmp_rank = lgb.LGBMRanker(n_estimators=300).fit(df[final_feats_without_stacking], 19 - df['着順'].clip(upper=18), group=final_groups)
tmp_class = lgb.LGBMClassifier(n_estimators=300).fit(df[final_feats_without_stacking], (df['着順'] <= 3).astype(int))
tmp_reg = lgb.LGBMRegressor(n_estimators=300).fit(df[final_feats_without_stacking], df['着順'])

scaler_tmp = MinMaxScaler()
all_r_p = scaler_tmp.fit_transform(tmp_rank.predict(df[final_feats_without_stacking]).reshape(-1, 1)).flatten()
all_c_p = tmp_class.predict_proba(df[final_feats_without_stacking])[:, 1]
all_reg_p = scaler_tmp.fit_transform((-tmp_reg.predict(df[final_feats_without_stacking])).reshape(-1, 1)).flatten()

all_batch_df = pd.DataFrame({'r_p': all_r_p, 'c_p': all_c_p, 'reg_p': all_reg_p})
df['Stacking_Score'] = meta_model.predict_proba(all_batch_df)[:, 1]

# 最終的な特徴量リストを確定
final_feats = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate', 'trainer_place_turf_dirt_rate', 'Stacking_Score']
print(final_feats)
print("最終モデルの学習中...")

final_m_rank = lgb.LGBMRanker(n_estimators=300).fit(df[final_feats], 19 - df['着順'].clip(upper=18), group=final_groups)
final_m_class = lgb.LGBMClassifier(n_estimators=300).fit(df[final_feats], (df['着順'] <= 3).astype(int))
final_m_reg = lgb.LGBMRegressor(n_estimators=300).fit(df[final_feats], df['着順'])

# ==========================================
# 【新規】本番推論用の最終的な騎手効率辞書を全期間から作成して保存
# ==========================================
print("本番推論用の最終騎手効率辞書を保存中...")
final_jockey_counts = df['騎手'].value_counts()
final_valid_jockeys = final_jockey_counts[final_jockey_counts >= 5].index

if len(final_valid_jockeys) > 0:
    final_jockey_stats = df[df['騎手'].isin(final_valid_jockeys)].groupby('騎手').agg({
        '着順': lambda x: (x <= 3).sum(),
        '単勝': lambda x: (1 / x).sum()
    })
    final_jockey_eff_dict = (final_jockey_stats['着順'] / final_jockey_stats['単勝'].replace(0, 1)).to_dict()
else:
    final_jockey_eff_dict = {}

joblib.dump(final_jockey_eff_dict, 'jockey_efficiency_final.pkl')

# 2. 最終モデルを保存
joblib.dump(final_m_rank, 'model_rank.pkl')
joblib.dump(final_m_class, 'model_class.pkl')
joblib.dump(final_m_reg, 'model_reg.pkl')
joblib.dump(final_feats, 'feature_names.pkl') 
joblib.dump(meta_model, 'meta_model.pkl')

print("推論用の正規化スケーラーを保存中...")
r_p_all = final_m_rank.predict(df[final_feats]).reshape(-1, 1)
c_p_all = final_m_class.predict_proba(df[final_feats])[:, 1].reshape(-1, 1)
reg_p_all = (-final_m_reg.predict(df[final_feats])).reshape(-1, 1)

scaler_rank = MinMaxScaler().fit(r_p_all)
scaler_class = MinMaxScaler().fit(c_p_all)
scaler_reg = MinMaxScaler().fit(reg_p_all)

joblib.dump(scaler_rank, 'scaler_rank.pkl')
joblib.dump(scaler_class, 'scaler_class.pkl')
joblib.dump(scaler_reg, 'scaler_reg.pkl')
print("スケーラーの保存が完了しました。")

# 3. 前処理用定数の保存
preprocessing_consts = {
    'mean_weight': df['馬体重'].mean(),
    'mean_past_rank': 7.5
}
joblib.dump(preprocessing_consts, 'preprocessing_consts.pkl')

print("すべてのモデル、定数、騎手効率辞書の保存が完了しました！")
print("全期間学習モデルと特徴量リストを保存しました！")