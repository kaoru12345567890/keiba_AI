import pandas as pd

# 1. クリーンになったデータを読み込む
df = pd.read_csv('processed_8re_data.csv', header=None)

# 2. 並び替え：年(1), 月(2), 日(3), 場所(5), レース番号(8) で整理
# 並び替えたデータを一旦別名で保存せず、指示通りに処理します
df_sorted = df.sort_values([1, 2, 3, 5, 8])

# 3. 並び替えたデータを 'processed_9_data.csv' として保存（上書き）
df_sorted.to_csv('processed_9_data.csv', index=False, header=False)

print("完了: データを並び替え、processed_9_data.csv を上書き保存しました。")