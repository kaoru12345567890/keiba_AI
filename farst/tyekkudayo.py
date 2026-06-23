import pandas as pd
import glob
import os

def check_all_files_columns():
    # NewDataフォルダ内の全CSVを検索
    all_files = glob.glob("NewData/**/*.csv", recursive=True)
    
    print(f"--- {len(all_files)} 個のファイルを調査します ---\n")
    
    # 列数ごとのカウント用辞書
    stats = {}
    
    for file in all_files:
        try:
            # 1行だけ読み込んで列数をチェック（高速化のため）
            df_head = pd.read_csv(file, header=None, nrows=1)
            num_cols = df_head.shape[1]
            
            # 統計を取る
            stats[num_cols] = stats.get(num_cols, 0) + 1
            
            # もし期待する「42列」以外ならファイル名を表示
            if num_cols != 42:
                print(f"[!] 異常検出: {file} | 列数: {num_cols}")
                
        except Exception as e:
            print(f"[X] 読み込み失敗: {file} | エラー: {e}")
            
    print("\n--- 調査結果サマリー ---")
    for count, files in stats.items():
        print(f"列数 {count} のファイル: {files} 個")

if __name__ == "__main__":
    check_all_files_columns()