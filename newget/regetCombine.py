import pandas as pd
import glob
import os

def merge_keiba_data(base_directory):
    # 統合データを保持するリスト
    all_data = []
    
    # 競馬場フォルダのリストを取得 (指定ディレクトリ直下の全サブフォルダを競馬場名とみなす)
    # パス例: C:\keiba_AI\regetData\新潟\2026.csv
    search_path = os.path.join(base_directory, "*", "2026.csv")
    files = glob.glob(search_path)
    print(os.path.exists(base_dir))
    print(f"見つかったファイル数: {len(files)}")
    
    for file in files:
        try:
            # 競馬場名をパスから抽出 (フォルダ名)
            stadium_name = os.path.basename(os.path.dirname(file))
            
            # CSV読み込み
            df = pd.read_csv(file)
            
            # 競馬場列を追加して識別できるようにする
            df['stadium'] = stadium_name
            
            all_data.append(df)
            print(f"読み込み完了: {stadium_name}")
            
        except Exception as e:
            print(f"エラー発生 ({file}): {e}")
            
    if all_data:
        # 全てを統合
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # 保存先 (基底ディレクトリと同じ場所に保存)
        output_path = os.path.join(r"C:\keiba_AI\newget", "all_2026_merged.csv")
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"統合完了: {output_path}")
    else:
        print("データが読み込まれませんでした。パスを確認してください。")

# 実行部
if __name__ == "__main__":
    # ユーザー様の環境に合わせてパスを指定してください
    base_dir = r"C:\keiba_AI\newget\regetData"
    merge_keiba_data(base_dir)