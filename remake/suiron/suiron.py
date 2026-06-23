import pandas as pd
import lightgbm as lgb
import os

# --- 【重要】学習モデルが要求する特徴量リスト ---
# today_race.csv は必ず以下の列名を含んでいる必要があります
feature_cols = [
    '年', '月', '日', '曜日', '場所', '回', '日目', 'レース目', 
    '天気', '馬場状態', 'レース条件', '芝ダート', '距離', '回り', '出走数', 
    '枠番', '馬番', '斤量', '騎手', '性別', '年齢', '所属', '調教師名', 
    'ペース', '馬体重', '体重増減', '過去出走回数', 
    '過去平均着順', '過去連対率', '過去複勝率', '過去平均上がり偏差値'
]

# 1. 今日の出走データ読み込み
if not os.path.exists('today_race.csv'):
    print("エラー: today_race.csv が見つかりません。")
else:
    today_df = pd.read_csv('today_race.csv')
    
    # モデルのロード
    model_files = {
        '勝率': 'model_is_1st.txt',
        '連対率': 'model_is_top2.txt',
        '複勝率': 'model_is_top3.txt'
    }
    
    # 2. 予測実行
    for name, file in model_files.items():
        if os.path.exists(file):
            booster = lgb.Booster(model_file=file)
            # feature_cols を使って予測
            today_df[name] = booster.predict(today_df[feature_cols].astype(float))
        else:
            print(f"警告: {file} が見つかりません。")

    # 3. レースごとにグループ化して上位10頭を抽出
    results = []
    # CSV内の 'レース目' 列を使って各レースごとに処理
    for race_num, group in today_df.groupby('レース目'):
        top10 = group.sort_values('勝率', ascending=False).head(10)
        results.append(top10)
    
    # 4. 全レースの結果を結合
    final_df = pd.concat(results)
    
    # 5. 保存と表示
    output_cols = ['レース目', '馬名', '枠番', '馬番', '勝率', '連対率', '複勝率']
    if 'オッズ' in final_df.columns:
        output_cols.append('オッズ')
        
    final_df[output_cols].to_csv('prediction_summary_all_races.csv', index=False, encoding='utf_8_sig')
    
    print("--- 全レース 予測確率一覧（上位10頭） ---")
    for race_num, group in final_df.groupby('レース目'):
        print(f"\n--- 第{race_num}レース ---")
        print(group[output_cols].to_string(index=False))