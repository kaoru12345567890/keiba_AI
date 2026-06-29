import pandas as pd
import re
import numpy as np

# ファイルパスの設定
INPUT_FILE = 'all_2026_merged.csv'
OUTPUT_FILE = 'renew.csv'

# 想定カラム構成
EXPECTED_COLUMNS = [
    "レースID", "年", "月", "日", "曜日", "場所", "回", "日目", "レース目", "レース名", 
    "天気", "馬場状態", "レース条件", "芝ダート", "距離", "回り", "出走馬数", "枠番", 
    "馬番", "人気", "馬名", "性別年齢", "斤量", "騎手", "タイム", "着差", "ペース", 
    "dummy1", "dummy2", "dummy3", "dummy4", "通過順", "上がり3F", "単勝", "着順", 
    "馬体重増減", "dummy5", "dummy6", "dummy7", "調教師", "馬主", "賞金", "dummy8"
]

def calculate_leg_style(row):
    passing_order = str(row['通過順'])
    n = float(row['出走馬数'])
    
    if passing_order == '**' or ('-' not in passing_order and not passing_order.isdigit()):
        return 0, 0.0

    pos = [int(p) for p in passing_order.replace('-', ' ').split()]
    if n <= 1: return 0, 0.0
    
    k = len(pos)
    if k == 4: weighted_sum = pos[0]*0.20 + pos[1]*0.35 + pos[2]*0.25 + pos[3]*0.15
    elif k == 3: weighted_sum = pos[0]*0.35 + pos[1]*0.45 + pos[2]*0.20
    elif k == 2: weighted_sum = pos[0]*0.60 + pos[1]*0.40
    else: weighted_sum = pos[0]
    
    a = (weighted_sum - 1) / (n - 1)
    
    # 脚質判定: 逃げ:1, 先行:2, 差し:3, 追込:4
    if a < 0.25: label = 1
    elif a < 0.5: label = 2
    elif a < 0.75: label = 3
    else: label = 4
    
    return label, round(a, 3)

def run_conversion(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        return f"エラー: {input_path} が見つかりません。"
    
    if len(df.columns) != len(EXPECTED_COLUMNS):
        return f"エラー: カラム数が一致しません。ファイル: {len(df.columns)}, 想定: {len(EXPECTED_COLUMNS)}"
    
    df.columns = EXPECTED_COLUMNS

    # 脚質計算の適用
    leg_results = df.apply(calculate_leg_style, axis=1, result_type='expand')
    df['脚質ラベル'] = leg_results[0]
    df['脚質スコア'] = leg_results[1]

    # 新規データフレーム作成
    new_df = pd.DataFrame()
    
    # マッピング
    new_df['レースID'] = df['レースID']
    new_df['年'] = df['年']
    new_df['月'] = df['月']
    new_df['日'] = df['日']
    new_df['曜日'] = df['曜日']
    new_df['場所'] = df['場所']
    new_df['回'] = df['回']
    new_df['日目'] = df['日目']
    new_df['レース目'] = df['レース目']
    new_df['レース名'] = df['レース名']
    new_df['天気'] = df['天気']
    new_df['馬場状態'] = df['馬場状態']
    new_df['レース条件'] = df['レース条件']
    new_df['芝ダート'] = df['芝ダート']
    new_df['距離'] = df['距離']
    new_df['回り'] = df['回り']
    new_df['出走数'] = df['出走馬数']
    new_df['着順'] = df['着順']
    new_df['枠番'] = df['枠番']
    new_df['馬番'] = df['馬番']
    new_df['馬名'] = df['馬名']
    new_df['性別年齢'] = df['性別年齢']
    new_df['斤量'] = df['斤量']
    new_df['騎手'] = df['騎手']
    new_df['タイム'] = df['タイム']
    new_df['着差'] = df['着差']
    new_df['ペース'] = df['ペース']
    new_df['通過順'] = df['通過順']
    new_df['上り3ハロン'] = df['上がり3F']
    new_df['単勝'] = df['単勝']
    new_df['人気'] = df['人気']
    
    # 体重・増減処理
    def split_weight(val):
        val = str(val)
        match = re.match(r'(\d+)\((.*)\)', val)
        return (float(match.group(1)), float(match.group(2))) if match else (None, 0.0)

    weights = df['馬体重増減'].apply(split_weight)
    new_df['馬体重'] = [x[0] for x in weights]
    new_df['体重増減'] = [x[1] for x in weights]
    
    # 調教師・所属・馬主・賞金
    trainer_col = df['調教師'].astype(str)
    new_df['所属'] = trainer_col.str.extract(r'\[(.*)\]')[0]
    new_df['調教師名'] = trainer_col.str.replace(r'\[.*\]', '', regex=True)
    new_df['馬主'] = df['馬主']
    new_df['賞金'] = df['賞金']
    
    # 性別・年齢
    new_df['性別'] = df['性別年齢'].str[0]
    new_df['年齢'] = df['性別年齢'].str[1:]
    
    # 脚質項目
    new_df['脚質スコア'] = df['脚質スコア']
    new_df['脚質ラベル'] = df['脚質ラベル']
    
    # 追加項目
    new_df['上がり偏差値'] = 0.0
    new_df['過去出走回数'] = 0
    new_df['過去平均着順'] = 0.0
    new_df['過去連対率'] = 0.0
    new_df['過去複勝率'] = 0.0
    new_df['過去平均上がり偏差値'] = 0.0
    
    # 日付順にソート（年、月、日）
    new_df = new_df.sort_values(by=['年', '月', '日']).reset_index(drop=True)
    
    new_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return f"{output_path} を作成し、日付順にソートしました。"

# 実行
result = run_conversion(INPUT_FILE, OUTPUT_FILE)
print(result)