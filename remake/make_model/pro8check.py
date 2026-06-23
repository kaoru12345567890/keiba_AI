import pandas as pd

# 1. データを読み込む（ヘッダーなし）
df = pd.read_csv('processed_8_data.csv', header=None)

# 元の行数を取得
original_count = len(df)

# 2. ゴミ行を特定して削除
# 0列目が「0」である行を除外
df_cleaned = df[df[0] != 0]

# 削除後の行数を取得し、差分を計算
new_count = len(df_cleaned)
deleted_count = original_count - new_count

# 3. クリーンなデータを保存
df_cleaned.to_csv('processed_8re_data.csv', index=False, header=False)

print(f"完了: {deleted_count} 個のゴミ行を削除しました。")
print(f"残りの行数: {new_count} 行")