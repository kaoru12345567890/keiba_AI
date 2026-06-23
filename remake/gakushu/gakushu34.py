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
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import re
import joblib

# 1. 初期ファイル読み込み
df = pd.read_csv('processed_10_data.csv', low_memory=False)

# 数字化処理（floatに変換）
numeric_cols = ['年', '月', '日', 'レース目', '出走数', '枠番', '馬番',
                '斤量', '着順', '距離', '単勝','過去平均着順', '過去出走回数',
                '過去連対率', '過去複勝率', '過去平均上がり偏差値', '年齢', '馬体重']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

df['馬体重'] = df['馬体重'].fillna(df['馬体重'].mean()) #平均場体重にする
df['過去平均着順'] = df['過去平均着順'].fillna(7.5) # 平均的な着順で埋める

# 性別を数値化
if '性別' in df.columns:
    gender_map = {'牡': 0, '牝': 1, 'セ': 2}
    df['性別_code'] = df['性別'].map(gender_map).fillna(3).astype(int)

# ========================================================
# 1. レース名から「クラス_ランク」と「新馬戦フラグ」を作る関数
# ========================================================
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
    
    for col in ['斤量', '過去平均着順', '人気']:
        if col in t_df.columns:
            t_df[f'{col}_rel'] = t_df.groupby('レースキー')[col].transform(lambda x: x - x.mean())
    return t_df