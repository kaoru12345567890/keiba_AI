import pandas as pd
import os

def combine_keiba_data():
    file1_path = r'C:\keiba_AI\remake\processed_10_data.csv'
    file2_path = r'C:\keiba_AI\newget\renew.csv'
    output_path = r'C:\keiba_AI\final\processed_data.csv'

    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print("ファイルが見つかりません。パスを確認してください。")
        return

    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 確実に「レースID」という列名を使って除外処理を行う場合
    target_col = 'レースID'
    
    # df2の中から、df1のレースIDリストに存在しない行だけを抽出
    df2_filtered = df2[~df2[target_col].isin(df1[target_col])]

    # 結合
    df_combined = pd.concat([df1, df2_filtered], ignore_index=True)

    df_combined.to_csv(output_path, index=False, encoding='utf_8_sig')
    
    print(f"処理が完了しました。")
    print(f"結合後の総行数: {len(df_combined)}行")

if __name__ == "__main__":
    combine_keiba_data()