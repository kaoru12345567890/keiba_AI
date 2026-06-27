# ==========================================
# 　　＜＜データの特徴量選定と前処理の方針＞＞
#-----------------------------------------
#　　　　＜ベースデータにあり、使える初期要素＞
# レースID,年,月,日,場所,回,日目,レース目,
# レース名,天気,馬場状態,レース条件,芝ダート,距離,
# 回り,出走数,着順,枠番,馬番,馬名,性別年齢,斤量,騎手,
# タイム,着差,ペース,通過順,上り3ハロン,単勝,人気,馬体重,体重増減,
# 所属,調教師名,馬主,賞金,脚質スコア,脚質ラベル,上がり偏差値,性別,年齢,
# 過去出走回数,過去平均着順,過去連対率,過去複勝率,過去平均上がり偏差値
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

# 1. 初期ファイル読み込み
print("データ読み込み中...")
df = pd.read_csv('processed_10_data.csv', low_memory=False)

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

df['体重増減'] = df['体重増減'].apply(clean_weight_diff)
df['体重増減'] = df['体重増減'].fillna(0.0)
    
print("馬体重の処理中...")
df['馬体重'] = df['馬体重'].fillna(df['馬体重'].mean()) #平均場体重にする
df['過去平均着順'] = df['過去平均着順'].fillna(7.5) # 平均的な着順で埋める

# 性別を数値化
print("性別の数値化処理中...")
if '性別' in df.columns:
    gender_map = {'牡': 0, '牝': 1, 'セ': 2}
    df['性別_code'] = df['性別'].map(gender_map).fillna(3).astype(int)

# ========================================================
# 1. レース名から「クラス_ランク」と「新馬戦フラグ」を作る関数
# ========================================================
print("クラス_ランクと新馬戦フラグの作成中...")
def get_class_rank_auto(row):
    # 万が一 NaN だった場合の対策
    if pd.isna(row['レース名']) or pd.isna(row['レース目']):
        return 0
    
    race_name = str(row['レース名'])
    race_num = int(row['レース目']) 
    
    if '新馬' in race_name or '未勝利' in race_name: 
        return 1
    elif '500万下' in race_name or '1勝クラス' in race_name: 
        return 2
    elif '1000万下' in race_name or '2勝クラス' in race_name: 
        return 3
    elif '1600万下' in race_name or '3勝クラス' in race_name: 
        return 4
    elif '第' in race_name and '回' in race_name:
        return 6  # 重賞
    else:
        return 5  # オープン

def apply_race_conditions(t_df):
    def get_weight_rule(cond):
        cond = str(cond)
        if '見習' in cond: return 4
        elif 'ハンデ' in cond: return 3
        elif '別定' in cond: return 2
        else: return 1

    t_df['斤量_ルール'] = t_df['レース条件'].apply(get_weight_rule)
    t_df['is_牝馬限定'] = t_df['レース条件'].astype(str).apply(lambda x: 1 if '牝' in x else 0)
    t_df['is_ハンデ戦'] = t_df['レース条件'].astype(str).apply(lambda x: 1 if 'ハンデ' in x else 0)
    return t_df

#特徴量作成
print("特徴量作成中...")
df['クラス_ランク'] = df.apply(get_class_rank_auto, axis=1)
df['is_新馬戦'] = df['レース名'].astype(str).apply(lambda x: 1 if '新馬' in x else 0)
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
    
    t_df['weight_ratio'] = t_df['馬体重'] / t_df['斤量'].replace(0, 1)
    t_df['weight_diff'] = t_df['馬体重'] - t_df['斤量'] * 8
    t_df['popularity_vs_ability'] = t_df['単勝'] / (t_df['過去平均着順'] + 1)
    t_df['weight_age_ratio'] = t_df['馬体重'] * t_df['年齢']
    t_df['jockey_trainer'] = t_df['騎手'].astype(str) + '_' + t_df['調教師名'].astype(str)
    t_df['レースキー'] = t_df['年'].astype(int).astype(str) + t_df['月'].astype(int).astype(str) + t_df['日'].astype(int).astype(str) + t_df['場所'].astype(str) + t_df['レース目'].astype(int).astype(str)
    
    for col in ['weight_ratio', '斤量', '過去平均着順', '単勝']:
        if col in t_df.columns:
            t_df[f'{col}_rel'] = t_df.groupby('レースキー')[col].transform(lambda x: x - x.mean())

    for i in [1, 2, 3, 4]:# 脚質ラベルの数値化（1,2,3,4）に基づいて、各レース内での頭数と割合を計算
        # そのレース（レースキー）の中に、脚質 i が何頭いるか
        t_df[f'脚質{i}_頭数'] = t_df.groupby('レースキー')['脚質ラベル'].transform(lambda x: (x == i).sum())
        
        # 出走数に対する割合（少頭数と多頭数の違いをなくすため）
        t_df[f'脚質{i}_割合'] = t_df[f'脚質{i}_頭数'] / t_df['出走数'].replace(0, 1)
    
    # <新しく追加した特徴量のリスト>
    df['騎手_競馬場_芝ダート'] = df['騎手'].astype(str) + '_' + df['場所'] + '_' + df['芝ダート']
    df['騎手_競馬場_芝ダート_回り'] = df['騎手_競馬場_芝ダート'] + '_' + df['回り']
    # 脚質は「脚質スコア」や「脚質コード」を使います
    df['騎手_競馬場_芝ダート_回り_脚質'] = (df['騎手_競馬場_芝ダート_回り'] + '_' + df['脚質ラベル'].astype(str))



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
        weight *= 3.0  # 函館のレースは重視
    elif '札幌' in place:
        weight *= 1.5  # 札幌のレースは中程度に重視
        
    # ② 直近3年の重み（最新トレンドの重視）
    if row['年'] >= (max_year - 2):  # 当年、前年、前々年
        weight *= 1.2
        
    # ③ クラスの重み（ノイズの多い下級条件を割引）
    # クラス_ランク: 1(新馬/未勝利), 2(1勝), 3(2勝), 4(3勝), 5(OP), 6(重賞)
    rank = row['クラス_ランク']
    if rank == 1:
        weight *= 0.7  # 新馬・未勝利はノイズが多いので30%オフ
    elif rank >= 5:
        weight *= 1.2  # OP・重賞は実力通りなので20%マシ
        
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
    '騎手_競馬場_芝ダート', '騎手_競馬場_芝ダート_回り','騎手_競馬場_芝ダート_回り_脚質'
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
print("統計辞書の作成中...")
# 最終的な統計量を計算
stats_dict = {
    'course_frame_rate': df.groupby('course_frame_key')['着順'].apply(lambda x: (x <= 3).mean()).to_dict(),
    'course_style_rate': df.groupby('course_style_key')['着順'].apply(lambda x: (x <= 3).mean()).to_dict()
}
joblib.dump(stats_dict, 'stats_dict.pkl')
print("統計辞書を保存しました。")

# ==========================================
# 4. 学習の実行（最終学習）
# ==========================================
print("最終モデルの学習開始...")

# 特徴量リストの定義（add_features等で生成した列を全て含む）
# '着順'などは学習に使わないので除外リストに入れる
drop_cols = [# 学習に使わない列のリスト
    # 予測対象（着順）や、学習に使わない列
    # ターゲット（正解ラベル）
    '着順', 
    # ID系（モデルが数字として意味を履き違える可能性があるもの）
    'レースID', 'レースキー', 'レース名', '馬名',
    # 重複・構造上の不要データ
    '年', '月', '日', '曜日', 'レース条件', 
    # 既に加工済み、または数値化済みの元の文字列データ
    '性別', '場所', '回り', '天気', '馬場状態', 
    '騎手', '所属', '調教師名', '脚質ラベル','ペース',
    # 評価には必要だが、学習には入れないもの
    'weight', "性別年齢", "タイム", "着差", "通過順", "馬主", "賞金",
    # ここに追加で除外したい列があれば追記
    '回', '日目', 'レース目', '斤量', '上り3ハロン', '単勝', '人気',
    '馬体重', '体重増減', '脚質スコア', '上がり偏差値'
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
years = sorted(df['年'].unique())

for target_year in tqdm(years, desc="年度別学習"):
    # データを分割
    train = df[df['年'] < target_year].copy()
    test = df[df['年'] == target_year].copy()
    if train.empty or test.empty: continue
    
    # 統計量の計算（必ずtrainのみから計算すること！）
    stats_frame_course = train.groupby('course_frame_key')['着順'].apply(lambda x: (x <= 3).mean())
    stats_course_style = train.groupby('course_style_key')['着順'].apply(lambda x: (x <= 3).mean())
    
    # testデータへの適用（fill_na(0)は未知のコース対策として必須）
    for t_df in [train, test]:
        t_df['course_frame_rate'] = t_df['course_frame_key'].map(stats_frame_course).fillna(0)
        t_df['course_style_rate'] = t_df['course_style_key'].map(stats_course_style).fillna(0)
    
    # 使用する特徴量（base + 新しい動的特徴量）
    current_feats = feat_base + ['course_frame_rate', 'course_style_rate']
    
    # モデルの学習
    # Rankerはgroupを指定することでレース単位の相対評価が可能になります
    groups_train = train.groupby('レースキー').size().to_numpy()
    
    m_rank = lgb.LGBMRanker(n_estimators=300).fit(train[current_feats], 19 - train['着順'].clip(upper=18), group=groups_train)
    m_class = lgb.LGBMClassifier(n_estimators=300).fit(train[current_feats], (train['着順'] <= 3).astype(int))
    m_reg = lgb.LGBMRegressor(n_estimators=300).fit(train[current_feats], train['着順'])
    
    importance_df = pd.DataFrame({
        'feature': current_feats,
        'importance': m_rank.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    print("\n=== 重要度トップ10（リーク確認用） ===")
    print(importance_df.head(10))

    # 予測（MinMaxScalerは学習データ全体の範囲でスケールするのが一般的）
    scaler = MinMaxScaler()
    r_p = scaler.fit_transform(m_rank.predict(test[current_feats]).reshape(-1, 1)).flatten()
    c_p = m_class.predict_proba(test[current_feats])[:, 1]
    
    # 回帰値は小さいほど着順が良いので負の数にしてMinMaxScalerを通す
    reg_p = scaler.fit_transform((-m_reg.predict(test[current_feats])).reshape(-1, 1)).flatten()
    
    test['Model_A'] = (r_p * 0.4) + (c_p * 0.3) + (reg_p * 0.3)
    test['Model_B'] = r_p
    test['Model_C'] = (c_p * 0.5) + (reg_p * 0.5)
    
    # 評価をループ内で実行し、進捗を確認する
    print(f"\n{target_year}年 評価結果:")
    print(evaluate_detailed(test, 'Model_A'))
    
    all_results.append(test)

# 全結果を統合
final = pd.concat(all_results)

# # 1. 全データで最終学習を行う
# print("--- 全期間データでの最終モデル学習を開始 ---")
# final_groups = df.groupby('レースキー').size().to_numpy()
# # 最終統計量での予測用に、全データに統計量を反映
# df['course_frame_rate'] = df['course_frame_key'].map(df.groupby('course_frame_key')['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)
# df['course_style_rate'] = df['course_style_key'].map(df.groupby('course_style_key')['着順'].apply(lambda x: (x <= 3).mean())).fillna(0)
# final_feats = feat_base + ['course_frame_rate', 'course_style_rate']

# final_m_rank = lgb.LGBMRanker(n_estimators=300).fit(df[final_feats], 19 - df['着順'].clip(upper=18), group=final_groups)
# final_m_class = lgb.LGBMClassifier(n_estimators=300).fit(df[final_feats], (df['着順'] <= 3).astype(int))
# final_m_reg = lgb.LGBMRegressor(n_estimators=300).fit(df[final_feats], df['着順'])

# # 2. 最終モデルを保存
# joblib.dump(final_m_rank, 'model_rank.pkl')
# joblib.dump(final_m_class, 'model_class.pkl')
# joblib.dump(final_m_reg, 'model_reg.pkl')
# joblib.dump(final_feats, 'feature_names.pkl') 

# print("全期間学習モデルと特徴量リストを保存しました！")