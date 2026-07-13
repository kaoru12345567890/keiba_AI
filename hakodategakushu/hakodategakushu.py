# ==========================================
# 　　＜＜データの特徴量選定と前処理の方針＞＞
#-----------------------------------------
#　　　　＜ベースデータにあり、使える初期要素＞
# レースID,年,月,日,場所,回,日目,レース目,
# レース名,天気,馬場状態,レース条件,芝ダート,距離,
# 回り,出走数,着順,枠番,馬番,馬名,性別年齢,斤量,騎手,
# タイム,着差,ペース,通過順,上り3ハロン,単勝,人気,馬体重,体重増減,
# 所属,調教師名,馬主,賞金,脚質スコア,脚質ラベル,上がり偏差値,性別,年齢,
# 過去出走回数,過去平均着順,過去連対率,過去複勝率,過去平均上がり偏差値,
#   ->増えたよまあ全部使おうかな
#  
#-----------------------------------------
#　　　　＜リークの可能性がある要素＞
# (着順),上り3ハロン,単勝,人気,上がり偏差値,タイム,ペース,
#-----------------------------------------
#　　　　＜今回のモデルで全く使わない要素＞
# 曜日,着差,通過順,上り3ハロン,上がり偏差値,人気,馬主,賞金,脚質スコア,性別年齢,タイム,ペース,
#-----------------------------------------
#　　　　＜今回のモデルで使う要素＞
# 年,月,日,場所,回,日目,レース目,天気,馬場状態,芝ダート,
# 回り,距離,出走数,枠番,馬番,斤量,(着順),馬体重,体重増減,
# 性別,年齢,騎手,所属,調教師名,脚質ラベル,単勝,
# 過去出走回数,過去平均着順,過去連対率,過去複勝率,過去平均上がり偏差値
#-----------------------------------------
# 　　　　＜使いたけど使い方がわからない要素＞
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

# # 性別を数値化
# print("性別の数値化処理中...")
# if '性別' in df.columns:
#     gender_map = {'牡': 3, '牝': 2, 'セ': 1}
#     df['性別_code'] = df['性別'].map(gender_map).fillna(0).astype(int)

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
    # 文字列結合は .str.cat を使用、条件分岐は np.select
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
# 文字列結合を高速化
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
        # 一時的な列として真geries値を保持
        col_name_i = f'is_temp_脚質{i}'
        t_df[col_name_i] = (t_df['脚質ラベル'] == i).astype(int)
        
        # グループごとに合計を計算
        t_df[f'脚質{i}_頭数'] = t_df.groupby('レースキー')[col_name_i].transform('sum')
        t_df[f'脚質{i}_割合'] = t_df[f'脚質{i}_頭数'] / t_df['出走数'].replace(0, 1)
        
        # 一時的な列を削除
        t_df.drop(columns=[col_name_i], inplace=True)
    
    # 同型ライバル頭数（高速化）
    t_df['同型ライバル頭数'] = 0
    for i in [1, 2, 3, 4]:
        mask = (t_df['脚質ラベル'] == i)
        t_df.loc[mask, '同型ライバル頭数'] = t_df.loc[mask, f'脚質{i}_頭数'] - 1
    
    # <新しく追加した特徴量のリスト>
    t_df['騎手_競馬場_芝ダート'] = t_df['騎手'].astype(str) + '_' + t_df['場所'] + '_' + t_df['芝ダート']
    # 脚質は「脚質ラベル」を使います
    #t_df['騎手_競馬場_芝ダート_回り_脚質'] = (t_df['騎手_競馬場_芝ダート_回り'] + '_' + t_df['脚質ラベル'].astype(str))
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
    #過去3レース分のレースランク、距離、芝ダート、着順、脚質が欲しい。→作った。
    #あと、前走からの日数があるといいかも？ →作った。
    #


    return t_df


# これを追加！
df = add_features(df)
# print()を使ってターミナルに表示する
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
    # クラス_ランク: 1(新馬/未勝利), 2(1勝), 3(2勝), 4(3勝), 5(OP), 6(重賞)
    rank = row['クラス_ランク']
    if rank == 1:
        weight *= 0.7  # 新馬・未勝利はノイズが多いので30%オフ
    elif rank >= 5:
        weight *= 1.3  # OP・重賞は実力通りなので20%マシ
    

    field = str(row['場所'])
    if '稍' in field:
        weight *= 1.2  # 馬場状態が稍重の場合は10%増し
    elif '重' in field:
        weight *= 1.4  # 馬場状態が重の場合は20%増し
    elif '不' in field:
        weight *= 1.6  # 馬場状態が不良の場合は30%増し

    # もしStakcing_Scoreが高ければ、学習時の影響力を1.5倍にする
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
    '騎手_競馬場_芝ダート',# '騎手_競馬場_芝ダート_回り','騎手_競馬場_芝ダート_回り_脚質', 
    'jockey_脚質', '場所_脚質', '馬場状態_脚質', '場所_芝ダート_馬場状態','天気_脚質', '天気_騎手',
    '調教師名_所属','調教師名_is_新馬戦','騎手_is_新馬戦','調教師名_is_ハンデ戦','騎手_is_ハンデ戦',
    'hakodate_jockey_trainer', '調教師名_is_牝馬限定','騎手_is_牝馬限定','調教師名_斤量_ルール',
    '騎手_斤量_ルール','距離_脚質_馬場状態','距離_脚質','騙馬年齢','牝馬年齢','牡馬年齢',
    '騙馬斤量_rel','牝馬斤量_rel','牡馬斤量_rel',
    '1走前_芝ダート','2走前_芝ダート','3走前_芝ダート',
    '1走前_着順','2走前_着順','3走前_着順'
]

encoders = {} # エンコーダーを保存する辞書
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
joblib.dump(encoders, 'label_encoders.pkl')
print("エンコーダーを保存しました。")

## ==========================================
# 評価関数の定義（詳細版）
## ==========================================

def evaluate_detailed(df_test, score_col):
    df_test = df_test.copy()
    # ランク算出
    df_test['rank'] = df_test.groupby('レースキー')[score_col].rank(ascending=False, method='min')
    
    def get_metrics(x):
        # 実際の上位馬
        actual_1st = x.loc[x['着順'] == 1, '着順'].values
        actual_top3 = x.loc[x['着順'] <= 3, '着順'].values
        
        # 予測上位馬の着順を取得
        r1 = x.loc[x['rank'] == 1, '着順'].values
        r2 = x.loc[x['rank'] == 2, '着順'].values
        r3 = x.loc[x['rank'] == 3, '着順'].values
        top4 = x.loc[x['rank'] <= 4, '着順'].values
        top5 = x.loc[x['rank'] <= 5, '着順'].values
        
        # 3連複BOX(4頭)の判定
        is_3renpuku_box4 = all([i in top4 for i in actual_top3]) if len(actual_top3) == 3 else False
        
        # 3連単BOX(5頭)の判定: 1〜3着が予測した上位5頭の中に全て含まれるか
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
            '3renpuku_box5': is_3rentan_box5 # 5頭BOXなら3連単もカバー範囲内
        })
    
    # グループ化して集計し、平均をとって％表示
    return df_test.groupby('レースキー', group_keys=False).apply(get_metrics, include_groups=False).mean() * 100

# ==========================================
# 3. 本番用統計辞書の作成と保存
# ==========================================
overall_3rd_rate = (df['着順'] <= 3).mean()

# スムージング用の関数を定義しておくと便利です
def calculate_smooth_rate(group):
    # サンプル数が少ない場合は全体の平均に寄せる
    return (group.sum() + 10 * overall_3rd_rate) / (len(group) + 10)

print("統計辞書の作成中...")

# 統計辞書作成
stats_dict = {
    'course_frame_rate': df.groupby('course_frame_key')['着順'].apply(lambda x: calculate_smooth_rate((x <= 3).astype(int))).to_dict(),
    'course_style_rate': df.groupby('course_style_key')['着順'].apply(lambda x: calculate_smooth_rate((x <= 3).astype(int))).to_dict(),
    'jockey_place_turf_dirt_rate': df.groupby('騎手_競馬場_芝ダート')['着順'].apply(lambda x: calculate_smooth_rate((x <= 3).astype(int))).to_dict(),
    'trainer_place_turf_dirt_rate': df.groupby(['調教師名', '場所', '芝ダート'])['着順'].apply(lambda x: calculate_smooth_rate((x <= 3).astype(int))).to_dict(),
    'baseline': overall_3rd_rate # ついでに保存しておくと推論時に便利！
}

joblib.dump(stats_dict, 'stats_dict.pkl')
print("統計辞書を保存しました。")

# ==========================================
# 4. 学習の実行（最終学習）
# ==========================================
print("モデルの学習開始...")

# 特徴量リストの定義（add_features等で生成した列を全て含む）
# '着順'などは学習に使わないので除外リストに入れる
drop_cols = [# 学習に使わない列のリスト
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

# 最終学習用データ
train_X = df[feats]
train_y = df['着順']
train_w = df['weight']
groups = df.groupby('レースキー').size().to_numpy()

# 学習に使う特徴量リストのベース（drop_colsで除外したもの以外）
feat_base = [c for c in df.columns if c not in drop_cols]

all_results = []
print("--- 統合適性スコアを含む学習パイプライン実行中 ---")
years = [y for y in sorted(df['年'].unique()) if y >= 2021]

# 全ての年度を回す前に、あらかじめ「各年度の予測値」を保存する箱を用意する
meta_train_features = [] # データフレームのリストで保持
meta_train_target = []

meta_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
meta_pkl_path = 'meta_model.pkl'

# dfにStacking_Score列がなければ作成（初期値0）
if 'Stacking_Score' not in df.columns:
    df['Stacking_Score'] = 0.0

#=========
#函館専用関数
# ========
def analyze_universal_conditions(local_df, target_year):
    """
    エンコーダーで可読化し、累計分析と出現割合（的中率）を表示する関数
    """
    # 1. エンコーダーの読み込みと可読化
    try:
        encoders = joblib.load('label_encoders.pkl')
        readable_df = local_df.copy()
        for col, le in encoders.items():
            if col in readable_df.columns:
                # 数字を元の文字列に戻す
                readable_df[col] = le.inverse_transform(readable_df[col])
    except FileNotFoundError:
        print("警告: label_encoders.pkl が見つかりません。数字のまま分析します。")
        readable_df = local_df.copy()

    # 2. 累計用のストレージ（関数属性）
    if not hasattr(analyze_universal_conditions, "cumulative_missed"):
        analyze_universal_conditions.cumulative_missed = []
        analyze_universal_conditions.cumulative_failed = []
        # 全体データを蓄積して的中率の分母にする
        analyze_universal_conditions.all_history_df = pd.DataFrame()

    # 3. 穴馬と外した上位馬の抽出
    median_score = readable_df['Model_A'].median()
    top3_pred = readable_df.sort_values(by='Model_A', ascending=False).groupby('レースID').head(5)
    
    failed = top3_pred[top3_pred['着順'] > 3]
    missed = readable_df[(readable_df['Model_A'] < median_score) & (readable_df['着順'] <= 3)]
    
    # 蓄積
    analyze_universal_conditions.cumulative_missed.append(missed)
    analyze_universal_conditions.cumulative_failed.append(failed)
    analyze_universal_conditions.all_history_df = pd.concat([analyze_universal_conditions.all_history_df, readable_df])
    
    # 4. 累計データフレーム作成
    all_missed = pd.concat(analyze_universal_conditions.cumulative_missed)
    all_failed = pd.concat(analyze_universal_conditions.cumulative_failed)

    #
    if '騎手' in readable_df.columns:
        # 期待値の合計 (1/オッズ) を騎手ごとに集計
        jockey_stats = readable_df.groupby('騎手').agg({
            '着順': lambda x: (x <= 3).sum(), # 複勝数
            '単勝': lambda x: (1 / x).sum() # 期待値合計
        })
        # 効率性 = 実績 / 期待値
        jockey_stats['efficiency'] = jockey_stats['着順'] / jockey_stats['単勝']
        
        print(f"\n=== 騎手の真の実力ランキング (期待値に対する効率) ===")
        # 効率が高い順（実力派）と低い順（たくさん乗ってるだけ）を表示
        print(jockey_stats.sort_values(by='efficiency', ascending=False).head(5))
    
    # 5. 分析レポート出力
    universal_cols = ['騎手', '距離_脚質', '調教師名', 'jockey_trainer', 'frame_ratio']
    
    print(f"\n=== 【{target_year}年】 函館開催 統計分析レポート ===")
    
    for col in universal_cols:
        if col in readable_df.columns:
            # --- 累計用データの計算 ---
            top_missed_cum = all_missed[col].value_counts().index[:5].tolist()
            top_failed_cum = all_failed[col].value_counts().index[:5].tolist()
            
            # --- 出現率の算出 ---
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
    """
    全体の評価、開催地別の評価、および外した馬の分析を行う関数
    """
    test_df['place_code'] = test_df['レースID'].astype(str).str[4:6]
    models = ['Model_A', 'Model_B', 'Model_C', 'Model_D' ]
    
    # 1. 全体の評価
    print(f"\n=== {target_year}年 全体評価 ===")
    for m in models:
        print(f"--- {m} ---")
        print(evaluate_detailed(test_df, m))
        
    # 2. 函館開催の評価と分析（コード '02'）
    target_code = '02'
    local_df = test_df[test_df['place_code'] == target_code].copy()
    
    if not local_df.empty:
        print(f"\n=== {target_year}年 函館開催（場所コード:{target_code}）評価 ===")
        for m in models:
            print(f"--- {m} ---")
            print(evaluate_detailed(local_df, m))
        
        # --- 答え合わせロジックを統合 ---
        print(f"\n=== 函館開催：予測分析レポート ===")
        
        # 予測スコアの高い順に並び替え
        local_df = local_df.sort_values(by=['レースID', 'Model_A'], ascending=[True, False])
        
        # 上位3頭の抽出と的中確認
        top3_pred = local_df.groupby('レースID').head(3).copy()
        top3_pred['is_correct'] = (top3_pred['着順'] <= 3).astype(int)
        
        # 外した事例と見落とした穴馬の抽出（表示用）
        failed_list = top3_pred[top3_pred['is_correct'] == 0]
        missed_list = local_df[(local_df['Model_A'] < local_df['Model_A'].median()) & (local_df['着順'] <= 3)]
        
        print(f"\n【上位予想だが3着以内外】\n{failed_list[['レースID', '馬名', 'Model_A', '着順']].head(5)}")
        print(f"\n【見落とした穴馬（スコア低めだが3着以内）】\n{missed_list[['レースID', '馬名', 'Model_A', '着順']].head(5)}")
        
        # 統計分析関数の呼び出し（ここを修正）
        analyze_universal_conditions(local_df, target_year)
        local_df = apply_jockey_boost(local_df)
        local_df = apply_final_correction(local_df)
        print(local_df)
    else:
        print(f"\n函館開催のデータは見つかりませんでした。")

def apply_jockey_boost(local_df):
    """
    騎手のefficiencyに基づき、シグモイド関数で滑らかな補正倍率を計算する
    """
    # 1. 騎手ごとのefficiencyを計算
    jockey_stats = local_df.groupby('騎手').agg({
        '着順': lambda x: (x <= 3).sum(),
        '単勝': lambda x: (1 / x).sum()
    })
    jockey_stats['efficiency'] = jockey_stats['着順'] / jockey_stats['単勝'].replace(0, 1)
    
    # 2. シグモイド関数による滑らかな重み付け
    # efficiencyの平均値を中央値（中心点x0）として設定
    x0 = jockey_stats['efficiency'].mean()
    # kは曲線の急峻さ（大きいほど閾値に近い動きになる）
    k = 0.5 
    
    # シグモイドで 0.5〜1.0 の値を生成し、それを倍率として活用
    # 補正倍率を 1.0 〜 1.2 の間で変動させる設計です
    def get_sigmoid_boost(eff):
        sigmoid_val = 1 / (1 + np.exp(-k * (eff - x0)))
        # 1.0〜1.2倍の間で調整（最小値1.0, 最大値1.2）
        return 1.0 + (sigmoid_val * 0.2)

    # 各騎手のefficiencyから倍率を計算
    boost_map = {
        jockey: get_sigmoid_boost(eff)
        for jockey, eff in jockey_stats['efficiency'].items()
    }
    
    # 3. 補正を適用
    local_df['Model_A'] = local_df['Model_A'] * local_df['騎手'].map(boost_map).fillna(1.0)
    
    return local_df

def apply_final_correction(local_df):
    """
    騎手の成績を自動集計し、効率が悪い騎手を自動で抽出してデバフをかける
    """
    if '騎手' not in local_df.columns or '着順' not in local_df.columns or '単勝' not in local_df.columns:
        return local_df

    # 1. 騎手ごとの成績を自動集計
    jockey_stats = local_df.groupby('騎手').agg({
        '着順': lambda x: (x <= 3).sum(),
        '単勝': lambda x: (1 / x).sum()
    })
    jockey_stats['efficiency'] = jockey_stats['着順'] / jockey_stats['単勝'].replace(0, 1)
    
    # 2. 自動判定の閾値を設定
    # 効率が極端に低い騎手（例: 効率0.3未満かつ、ある程度の母数がある騎手）を自動抽出
    # ※集計回数が少ないと誤差が大きいため、騎乗回数も考慮するとより正確です
    jockey_counts = local_df['騎手'].value_counts()
    
    # 騎乗回数が5回以上あり、かつ効率が0.3以下の騎手を「カス騎手」として自動認定
    bad_jockeys = jockey_stats[
        (jockey_stats['efficiency'] < 0.3) & 
        (jockey_stats.index.map(jockey_counts) >= 5)
    ].index.tolist()
    
    # 3. 補正適用
    # 実力派（効率2.0以上）へのボーナス
    high_perf_jockeys = jockey_stats[jockey_stats['efficiency'] >= 2.0].index
    local_df.loc[local_df['騎手'].isin(high_perf_jockeys), 'Model_A'] *= 1.1
    
    # 自動抽出されたデバフ対象へのペナルティ
    if bad_jockeys:
        print(f"\n[自動デバフ] 以下の騎手を低効率として補正対象に設定: {bad_jockeys}")
        local_df.loc[local_df['騎手'].isin(bad_jockeys), 'Model_A'] *= 0.85
    
    return local_df


for target_year in tqdm(years, desc="年度別学習"):
    # 2010年から「テストする年の前年まで」をすべて学習データにする
    train = df[(df['年'] >= 2010) & (df['年'] < target_year)].copy()
    test = df[df['年'] == target_year].copy()
    
    if train.empty or test.empty: continue

    df.loc[test.index, 'Stacking_Score'] = test['Stacking_Score']

    # 統計量の計算（必ずtrainのみから計算すること！）
    stats_frame_course = train.groupby('course_frame_key')['着順'].apply(lambda x: (x <= 3).mean())
    stats_course_style = train.groupby('course_style_key')['着順'].apply(lambda x: (x <= 3).mean())
    stats_jockey_place_turf_dirt = train.groupby('騎手_競馬場_芝ダート')['着順'].apply(lambda x: (x <= 3).mean())
    stats_trainer_place_turf_dirt = train.groupby(['調教師名', '場所', '芝ダート'])['着順'].apply(lambda x: (x <= 3).mean())

    baseline_rate = (train['着順'] <= 3).mean()

    # testデータへの適用
    for t_df in [train, test]:
        t_df['course_frame_rate'] = t_df['course_frame_key'].map(stats_frame_course).fillna(baseline_rate)
        t_df['course_style_rate'] = t_df['course_style_key'].map(stats_course_style).fillna(baseline_rate)
        t_df['jockey_place_turf_dirt_rate'] = t_df['騎手_競馬場_芝ダート'].map(stats_jockey_place_turf_dirt).fillna(baseline_rate)
        # 調教師の集計も apply から map に変えると高速です
        t_df['trainer_key'] = list(zip(t_df['調教師名'], t_df['場所'], t_df['芝ダート']))
        t_df['trainer_place_turf_dirt_rate'] = t_df['trainer_key'].map(stats_trainer_place_turf_dirt).fillna(baseline_rate)

    # 使用する特徴量（base + 新しい動的特徴量）
    current_feats = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate', 'trainer_place_turf_dirt_rate', 'Stacking_Score']

    # モデルの学習
    # Rankerはgroupを指定することでレース単位の相対評価が可能になります
    train_weights = train.apply(calculate_complex_weight, axis=1).values

    # 2. モデルの学習時に sample_weight を引数に追加
    groups_train = train.groupby('レースキー').size().to_numpy()
    
    # LGBMRanker
    m_rank = lgb.LGBMRanker(n_estimators=300).fit(
        train[current_feats], 
        19 - train['着順'].clip(upper=18), 
        group=groups_train,
        sample_weight=train_weights # ここで適用
    )
    
    # LGBMClassifier
    m_class = lgb.LGBMClassifier(n_estimators=300).fit(
        train[current_feats], 
        (train['着順'] <= 3).astype(int),
        sample_weight=train_weights # ここでも適用
    )
    
    # LGBMRegressor (回帰モデルにも必要に応じて)
    m_reg = lgb.LGBMRegressor(n_estimators=300).fit(
        train[current_feats], 
        train['着順'],
        sample_weight=train_weights # ここでも適用
    )
    
    importance_df = pd.DataFrame({
        'feature': current_feats,
        'importance': m_rank.feature_importances_
    }).sort_values(by='importance', ascending=False)


    
    print("\n=== 重要度トップ20（リーク確認用） ===")
    print(importance_df.head(20))
    # 下位20個を表示（削除候補を見つけるために確認）
    print("\n=== 重要度下位20（断捨離候補） ===")
    print(importance_df.tail(20).to_string())

    # 予測（MinMaxScalerは学習データ全体の範囲でスケールするのが一般的）
    scaler = MinMaxScaler()
    r_p = scaler.fit_transform(m_rank.predict(test[current_feats]).reshape(-1, 1)).flatten()
    c_p = m_class.predict_proba(test[current_feats])[:, 1]
    # 回帰値は小さいほど着順が良いので負の数にしてMinMaxScalerを通す
    reg_p = scaler.fit_transform((-m_reg.predict(test[current_feats])).reshape(-1, 1)).flatten()
    
    # 【追加】メタ学習用の特徴量を蓄積（ベースモデルの予測値3つ）
    batch_df = pd.DataFrame({'r_p': r_p, 'c_p': c_p, 'reg_p': reg_p})
    meta_train_features.append(batch_df)
    meta_train_target.append((test['着順'] <= 3).astype(int))

    # --- ここがポイント：ループ内でメタモデルを逐次更新 ---
    X_meta = pd.concat(meta_train_features, ignore_index=True)
    y_meta = pd.concat(meta_train_target, ignore_index=True)
    
    # 2年目以降なら、過去の蓄積データでメタモデルを再学習・上書き
    if years.index(target_year) > 0: 
        meta_model.fit(X_meta, y_meta)
        joblib.dump(meta_model, meta_pkl_path) # モデルを上書き保存

        # 学習済みモデルで今すぐ補正スコアを計算し、dfに書き込む
        test['Stacking_Score'] = meta_model.predict_proba(batch_df)[:, 1]
        df.loc[test.index, 'Stacking_Score'] = test['Stacking_Score']

    test['Model_A'] = (r_p * 0.4) + (c_p * 0.3) + (reg_p * 0.3)
    test['Model_B'] = (r_p * 0.2) + (c_p * 0.6) + (reg_p * 0.2)
    test['Model_C'] = (c_p * 0.5) + (reg_p * 0.5)
    test['Model_D'] = (c_p)
    
    # 評価をループ内で実行し、進捗を確認する
    evaluate_and_print_results(test, target_year)


    all_results.append(test)

# 全結果を統合
final = pd.concat(all_results)

# ==========================================
# 【新規追加】メタモデル（スタッキング）の学習
# ==========================================
print("メタモデル（スタッキング）の学習を開始...")

X_meta = pd.concat(meta_train_features, ignore_index=True)
y_meta = pd.concat(meta_train_target, ignore_index=True)

# メタモデルを定義
meta_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
meta_model.fit(X_meta, y_meta)

print("メタモデルの学習が完了しました。")

# 【おまけ】最終的なスタッキング予測値の算出
# これを各年度のtestデータ（finalの中）に適用して、新たな予測スコア列を作る
meta_probs = meta_model.predict_proba(X_meta)[:, 1]
final['Stacking_Score'] = meta_probs

# ==========================================
# メタモデルの学習結果（重要度）の可視化
# ==========================================
print("\n=== メタモデルの判断基準（重要度） ===")
# メタモデルが、どの予測値を重視したかを表示
meta_importance = pd.DataFrame({
    'feature': ['r_p', 'c_p', 'reg_p'],
    'importance': meta_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print(meta_importance)

# 補足として、メタモデルの予測傾向も少し見ておきましょう
print("\n=== メタモデルの予測スコアの分布（期待値など） ===")
print(pd.Series(meta_probs).describe())
print("全期間のデータにスタッキングスコアを付与しました。")

# --- ここを既存のスタッキング学習の下に追加します ---
# メタモデルの予測値を全データに対して算出
final_meta_probs = meta_model.predict_proba(X_meta)[:, 1]

# dfに 'Stacking_Score' として結合
# ※dfのインデックスとmeta_trainの結果が一致している必要があります
df['Stacking_Score'] = final_meta_probs

# 1. 全データで最終学習を行う
print("--- 全期間データでの最終モデル学習を開始 ---")
final_groups = df.groupby('レースキー').size().to_numpy()
df['trainer_key'] = list(zip(df['調教師名'], df['場所'], df['芝ダート']))
# 最終統計量での予測用に、全データに統計量を反映
df['course_frame_rate'] = df['course_frame_key'].map(df.groupby('course_frame_key')['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)
df['course_style_rate'] = df['course_style_key'].map(df.groupby('course_style_key')['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)
df['jockey_place_turf_dirt_rate'] = df['騎手_競馬場_芝ダート'].map(stats_jockey_place_turf_dirt).fillna(0)
df['trainer_place_turf_dirt_rate'] = df['trainer_key'].map(stats_trainer_place_turf_dirt).fillna(0)

final_feats = feat_base + ['course_frame_rate', 'course_style_rate', 'jockey_place_turf_dirt_rate', 'trainer_place_turf_dirt_rate', 'Stacking_Score']
print(final_feats)
print("最終モデルの学習中...")

final_m_rank = lgb.LGBMRanker(n_estimators=300).fit(df[final_feats], 19 - df['着順'].clip(upper=18), group=final_groups)
final_m_class = lgb.LGBMClassifier(n_estimators=300).fit(df[final_feats], (df['着順'] <= 3).astype(int))
final_m_reg = lgb.LGBMRegressor(n_estimators=300).fit(df[final_feats], df['着順'])

# 2. 最終モデルを保存
joblib.dump(final_m_rank, 'model_rank.pkl')
joblib.dump(final_m_class, 'model_class.pkl')
joblib.dump(final_m_reg, 'model_reg.pkl')
joblib.dump(final_feats, 'feature_names.pkl') 
joblib.dump(meta_model, 'meta_model.pkl')

print("推論用の正規化スケーラーを保存中...")

# 1. 最終モデルを用いて全データで予測値（ベースモデルの出力）を出す
r_p_all = final_m_rank.predict(df[final_feats]).reshape(-1, 1)
c_p_all = final_m_class.predict_proba(df[final_feats])[:, 1].reshape(-1, 1)
reg_p_all = (-final_m_reg.predict(df[final_feats])).reshape(-1, 1) # 回帰値は符号反転

# 2. それぞれのスケーラーを作成し、学習データでfitして保存する
# それぞれの予測値の分布が異なるため、個別にスケーラーを作るのが安全です
scaler_rank = MinMaxScaler().fit(r_p_all)
scaler_class = MinMaxScaler().fit(c_p_all)
scaler_reg = MinMaxScaler().fit(reg_p_all)

# スケーラーを個別に保存
joblib.dump(scaler_rank, 'scaler_rank.pkl')
joblib.dump(scaler_class, 'scaler_class.pkl')
joblib.dump(scaler_reg, 'scaler_reg.pkl')

print("スケーラーの保存が完了しました。")

# 3. 前処理用定数の保存（推論時に同じ値を使うために重要！）
# 学習時に使用した固定値を辞書にする
preprocessing_consts = {
    'mean_weight': df['馬体重'].mean(),
    'mean_past_rank': 7.5
}
joblib.dump(preprocessing_consts, 'preprocessing_consts.pkl')

print("すべてのモデルと定数を保存しました！")

print("全期間学習モデルと特徴量リストを保存しました！")