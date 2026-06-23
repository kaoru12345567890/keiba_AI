import pandas as pd
import re

# 1. データの読み込み
df = pd.read_csv('processed_10_data.csv', header=None, low_memory=False)
col_names = [
    'レースID', '年', '月', '日', '曜日', '場所', '回', '日目', 'レース目', 'レース名', 
    '天気', '馬場状態', 'レース条件', '芝ダート', '距離', '回り', '出走数', '着順', 
    '枠番', '馬番', '馬名', '性別年齢', '斤量', '騎手', 'タイム', '着差', 'ペース', 
    '通過順', '上り3ハロン', '単勝', '人気', '馬体重', '体重増減', '所属', '調教師名', '馬主', 
    '賞金', '脚質スコア', '脚質ラベル', '上がり偏差値', '性別', '年齢', '過去出走回数', 
    '過去平均着順', '過去連対率', '過去複勝率', '過去平均上がり偏差値'
]
df.columns = col_names
df = df[df['年'] != '年'].copy()

# 2. 前処理
def clean_rank(val):
    val = str(val)
    res = re.sub(r'\D', '', val)
    return int(res) if res != '' else 0

df['着順'] = df['着順'].apply(clean_rank)
# 人気列を数値化（数値以外や欠損は排除）
df['人気'] = pd.to_numeric(df['人気'], errors='coerce')
df = df.dropna(subset=['人気'])

# 3. 集計
# 人気1～3位に限定
target_pop = df[df['人気'].isin([1, 2, 3])].copy()

# 1着・2着・3着の判定フラグ作成
target_pop['is_1st'] = (target_pop['着順'] == 1).astype(int)
target_pop['is_2nd'] = (target_pop['着順'] == 2).astype(int)
target_pop['is_3rd'] = (target_pop['着順'] == 3).astype(int)
target_pop['is_top3'] = (target_pop['着順'] <= 3).astype(int)

# グループ化して平均（＝確率）を出す
pop_stats = target_pop.groupby('人気').agg({
    'is_1st': 'mean',
    'is_2nd': 'mean',
    'is_3rd': 'mean',
    'is_top3': 'mean'
})

# パーセント表示で見やすくする
print("\n=== 人気別：着順確率（勝率・連対率・複勝率） ===")
print((pop_stats * 100).round(2).astype(str) + '%')

# カウントも出力
print("\n=== 人気別：実際の結果（着順カウント） ===")
pivot = pd.pivot_table(target_pop, values='馬番', index='人気', columns='着順', aggfunc='count', fill_value=0)
print(pivot[[1, 2, 3]])