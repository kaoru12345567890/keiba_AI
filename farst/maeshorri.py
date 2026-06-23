import csv

# 1. 設定：読み込むファイルと保存するファイル
input_file = 'master_data.csv'
output_file = 'master_data_cleaned.csv'

# 2. 設定：消したい列のインデックス（0から始まる番号）
# ここに消したい列番号を並べてください
ignore_indices = [36, 37, 38, 39, 40] 

# 3. データを読み込む
data_list = []
with open(input_file, mode='r', encoding='utf_8_sig') as file:
    reader = csv.reader(file)
    for row in reader:
        data_list.append(row)

# 4. データを処理する
cleaned_data = []
for row in data_list:
    new_row = []
    # 各行のすべての列をチェック
    for i in range(len(row)):
        # 指定した列番号なら飛ばす(continue)
        if i in ignore_indices:
            continue
        # 不要でないなら新しいリストに追加
        new_row.append(row[i])
    
    # 整形後の行を追加
    cleaned_data.append(new_row)

# 5. 整形したデータを新しいCSVとして保存
with open(output_file, mode='w', encoding='utf_8_sig', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(cleaned_data)

print(f"処理完了！ {output_file} を確認してください。")