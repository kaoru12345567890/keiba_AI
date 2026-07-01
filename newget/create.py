import pandas as pd
import re
import numpy as np
from tqdm import tqdm

# --- 設定 ---
INPUT_FILE = r'C:\keiba_AI\newget\all_2026_merged.csv'
OUTPUT_FILE = r'C:\keiba_AI\newget\renew.csv'
ERROR_LOG_FILE = r'C:\keiba_AI\newget\error_log.txt'

# 想定カラム構成（計43列）
EXPECTED_COLUMNS = [
    "レースID", "年", "月", "日", "曜日", "場所", "回", "日目", "レース目", "レース名", 
    "天気", "馬場状態", "レース条件", "芝ダート", "距離", "回り", "出走馬数", "枠番", 
    "馬番", "人気", "馬名", "性別年齢", "斤量", "騎手", "タイム", "着差", "ペース", 
    "dummy1", "dummy2", "dummy3", "dummy4", "通過順", "上がり3F", "単勝", "着順", 
    "馬体重増減", "dummy5", "dummy6", "dummy7", "調教師", "馬主", "賞金", "dummy8"
]

def run_conversion():
    # 1. 読み込み
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        return f"エラー: {INPUT_FILE} が見つかりません。"
    
    if len(df.columns) != len(EXPECTED_COLUMNS):
        return f"エラー: カラム数が一致しません。ファイル: {len(df.columns)}, 想定: {len(EXPECTED_COLUMNS)}"
    
    df.columns = EXPECTED_COLUMNS
    # 枠番カラムから「除」「取」「中」を含む行を削除
    df['枠番'] = df['枠番'].astype(str)
    df['芝ダート'] = df['芝ダート'].astype(str)
    exclude_mask = (df['枠番'].str.contains('除|取|中', na=False)) | (df['芝ダート'].str.contains('障', na=False))
    print(f"削除対象のデータ数: {exclude_mask.sum()} 件")
    df = df[~exclude_mask].copy()
    
    # 着順を数値化（念のため）
    df['枠番'] = pd.to_numeric(df['枠番'], errors='coerce')
    df['上がり3F'] = pd.to_numeric(df['上がり3F'], errors='coerce')

    # 2. 統計値事前計算（上がり偏差値用）
    race_stats = df.groupby('レースID')['上がり3F'].agg(['mean', 'std'])
    error_list = []

    # 3. 特徴量計算（脚質・偏差値）
    def process_row(row):
        # --- 脚質計算 ---
        passing_order = str(row['通過順'])
        n = float(row['出走馬数'])
        score, label_id = 0.5, 0
        
        try:
            pos = [int(p) for p in passing_order.replace('-', ' ').split()]
            if n > 1:
                weighted_sum = pos[0] if len(pos) == 1 else (pos[0]*0.6 + pos[1]*0.4)
                a = (weighted_sum - 1) / (n - 1)
                score = round(a, 3)
                if a < 0.25: label_id = 1
                elif a < 0.5: label_id = 2
                elif a < 0.75: label_id = 3
                else: label_id = 4
        except:
            pass
            
        # --- 上がり偏差値計算 ---
        dev = 50.0
        try:
            agari = row['上がり3F']
            stats = race_stats.loc[row['レースID']]
            if not pd.isna(agari) and stats['std'] > 0:
                dev = round(50 + 10 * ((stats['mean'] - agari) / stats['std']), 2)
        except Exception as e:
            error_list.append(f"RaceID: {row['レースID']}, 馬名: {row['馬名']}, Error: {str(e)}")
            
        return pd.Series([score, label_id, dev])

    print("特徴量を計算中...")
    tqdm.pandas()
    results = df.progress_apply(process_row, axis=1)
    df['脚質スコア'], df['脚質ラベル'], df['上がり偏差値'] = results[0], results[1], results[2]

    # 4. 新規データフレーム構築
    new_df = pd.DataFrame()
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
    new_df['人気'] = df['着順']
    new_df['着順'] = df['枠番']
    new_df['枠番'] = df['馬番']
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
    new_df['馬番'] = df['人気']
    
    # 馬体重分解
    weights = df['馬体重増減'].apply(lambda val: re.match(r'(\d+)\((.*)\)', str(val)))
    new_df['馬体重'] = [float(m.group(1)) if m else None for m in weights]
    new_df['体重増減'] = [float(m.group(2)) if m else 0.0 for m in weights]
    
    # 調教師・所属
    new_df['所属'] = df['調教師'].str.extract(r'\[(.*)\]')[0]
    new_df['調教師名'] = df['調教師'].str.replace(r'\[.*\]', '', regex=True)
    new_df['馬主'] = df['馬主']
    new_df['賞金'] = df['賞金']
    
    # 性別数値変換（セ=1, 牝=2, 牡=3, その他=0）
    sex_raw = df['性別年齢'].str[0]
    sex_map = {'セ': 1, '牝': 2, '牡': 3}
    new_df['性別'] = sex_raw.map(sex_map).fillna(0).astype(int)
    new_df['年齢'] = df['性別年齢'].str[1:]
    
    new_df['脚質スコア'] = df['脚質スコア']
    new_df['脚質ラベル'] = df['脚質ラベル']
    new_df['上がり偏差値'] = df['上がり偏差値']
    
    # 過去データ系（初期値）
    for col in ['過去出走回数', '過去平均着順', '過去連対率', '過去複勝率', '過去平均上がり偏差値']:
        new_df[col] = 0.0

    # 順番を指定したリスト
    column_order = [
        "レースID", "年", "月", "日", "曜日", "場所", "回", "日目", "レース目", "レース名", 
        "天気", "馬場状態", "レース条件", "芝ダート", "距離", "回り", "出走数", "着順", 
        "枠番", "馬番", "馬名", "性別年齢", "斤量", "騎手", "タイム", "着差", "ペース", 
        "通過順", "上り3ハロン", "単勝", "人気", "馬体重", "体重増減", "所属", "調教師名", 
        "馬主", "賞金", "脚質スコア", "脚質ラベル", "上がり偏差値", "性別", "年齢", 
        "過去出走回数", "過去平均着順", "過去連対率", "過去複勝率", "過去平均上がり偏差値"
    ]
    
    # 指定した順序で列を並べ替え
    new_df = new_df[column_order]
    
    # 5. ソートして保存
    new_df = new_df.sort_values(by=['年', '月', '日', '場所', 'レース目']).reset_index(drop=True)
    new_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    # エラーログ処理
    if error_list:
        with open(ERROR_LOG_FILE, 'w', encoding='utf-8') as f:
            for err in error_list: f.write(err + '\n')
        print(f"[!] {len(error_list)}件のエラー。{ERROR_LOG_FILE}を確認してください。")
    
    return f"完了: {OUTPUT_FILE} を作成しました。"

if __name__ == "__main__":
    print(run_conversion())