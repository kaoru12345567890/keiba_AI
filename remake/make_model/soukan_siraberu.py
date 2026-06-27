import pandas as pd

# 1. データの読み込みと前処理
df = pd.read_csv("processed_10_data.csv", low_memory=False)
df_shiba = df[df['芝ダート'] == '芝'].copy()

# 着順の数値化と欠損値の除外（前回同様のノイズ対策）
df_shiba['着順'] = pd.to_numeric(df_shiba['着順'], errors='coerce')
df_shiba = df_shiba.dropna(subset=['着順'])

# 2. ベースとなる「函館」のデータを作成
df_hakodate = df_shiba[df_shiba['場所'] == '函館'].groupby('馬名')['着順'].mean().reset_index()
df_hakodate.rename(columns={'着順': '函館_平均着順'}, inplace=True)

# 3. 函館以外の各競馬場との相関を計算する
results = []
# データ内に存在するすべての競馬場を取得し、函館と札幌を除外
all_courses = df_shiba['場所'].unique()
target_courses = [course for course in all_courses if course not in ['函館']]

for course in target_courses:
    # 対象の競馬場のデータを集計
    df_course = df_shiba[df_shiba['場所'] == course].groupby('馬名')['着順'].mean().reset_index()
    df_course.rename(columns={'着順': f'{course}_平均着順'}, inplace=True)
    
    # 函館のデータと結合 (共通して走った馬を抽出)
    df_merged = pd.merge(df_hakodate, df_course, on='馬名', how='inner')
    
    # 共通して走った馬が少なすぎると相関がブレるため、ある程度（例: 20頭以上）いる場合のみ計算
    common_horses = len(df_merged)
    if common_horses >= 20:
        correlation = df_merged['函館_平均着順'].corr(df_merged[f'{course}_平均着順'])
        
        results.append({
            '競馬場': course,
            '相関係数': correlation,
            '共通出走頭数': common_horses
        })

# 4. 結果をデータフレーム化し、相関係数が高い順（1に近い順）に並び替えて表示
df_results = pd.DataFrame(results)
if not df_results.empty:
    df_results = df_results.sort_values(by='相関係数', ascending=False).reset_index(drop=True)
    print("【函館競馬場と各競馬場の平均着順の相関（芝）】")
    print(df_results)
else:
    print("計算できる十分なデータがありませんでした。")