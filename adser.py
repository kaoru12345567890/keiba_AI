import pandas as pd

# CSVファイルの読み込む
df = pd.read_csv('processed_10_data.csv')

race_name_col = 'レース名'
race_cond_col = 'レース条件'

# テキストファイルとして保存
with open('race_data_check.txt', 'w', encoding='utf-8') as f:
    f.write("=========================================\n")
    f.write(f"【{race_name_col}】全種類一覧（出現回数順）\n")
    f.write("=========================================\n")
    # すべて書き出すために head() をつけずに文字列化
    f.write(df[race_name_col].value_counts().to_string())
    
    f.write("\n\n" + "="*41 + "\n")
    f.write(f"【{race_cond_col}】全種類一覧（出現回数順）\n")
    f.write("=========================================\n")
    f.write(df[race_cond_col].value_counts().to_string())

print("調査完了！「race_data_check.txt」というファイルにすべて保存しました。")