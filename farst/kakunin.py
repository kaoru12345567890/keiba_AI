import csv

def filter_csv(input_file, output_file, delimiter=','):
    """
    指定されたCSVファイルから、列数が3以上の行のみを抽出して保存します。
    """
    try:
        with open(input_file, mode='r', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile, delimiter=delimiter)
            
            with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile, delimiter=delimiter)
                
                count_removed = 0
                count_kept = 0
                
                for row in reader:
                    # 要素数が2以下の場合はスキップ（削除対象）
                    if len(row) <= 2:
                        count_removed += 1
                        continue
                    
                    # 要素数が3以上の場合は書き込み
                    writer.writerow(row)
                    count_kept += 1
        
        print(f"処理が完了いたしました。")
        print(f"残した行数: {count_kept}")
        print(f"削除した行数: {count_removed}")
        print(f"保存先: {output_file}")
        
    except Exception as e:
        print(f"エラーが発生いたしました: {e}")

# ファイルパスを指定して実行
# ファイルが同階層にある場合はファイル名のみで問題ございません
input_csv = 'horses_raw_database_62.csv'
output_csv = 'filtered_horses_database.csv'

# 区切り文字がカンマでない場合（タブなど）は delimiter='\t' に変更してください
filter_csv(input_csv, output_csv, delimiter=',')