import pandas as pd

# CSVファイルの読み込む
df = pd.read_csv('processed_10_data.csv')

race_id_col = 'レースID'
race_name_col = 'レース名'
race_cond_col = 'レース条件'

# 💡 修正のキモ：レースIDを基準にして、レースごとの固有情報だけを抜き出す（重複排除）
# これにより、1レース1行のデータに圧縮されます
race_unique_df = df[[race_id_col, race_name_col, race_cond_col]].drop_duplicates()

# テキストファイルとして保存
with open('race_data_check.txt', 'w', encoding='utf-8') as f:
    f.write("=========================================\n")
    f.write(f"【{race_name_col}】全種類一覧（純粋なレース数順）\n")
    f.write("=========================================\n")
    # 重複排除したデータからカウント
    f.write(race_unique_df[race_name_col].value_counts().to_string())
    
    f.write("\n\n" + "="*41 + "\n")
    f.write(f"【{race_cond_col}】全種類一覧（純粋なレース数順）\n")
    f.write("=========================================\n")
    # 重複排除したデータからカウント
    f.write(race_unique_df[race_cond_col].value_counts().to_string())

print("調査完了！レース重複を排除し「race_data_check.txt」に正しく保存しました。")