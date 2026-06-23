import pandas as pd

def extract_unique_horses():
    try:
        # 1. マスターデータを読み込む
        # ※encoding='utf_8_sig' は、日本語が含まれるCSVを扱う際のおまじないです
        df = pd.read_csv("master_data.csv", encoding='utf_8_sig')
        
        # 2. '馬名' 列から重複を削除したリストを作成
        # drop_duplicates() で重複を消し、reset_index() で番号を振り直します
        unique_horses = df[['馬名']].drop_duplicates().reset_index(drop=True)
        
        # 3. CSVファイルに保存
        output_file = "unique_horse_list.csv"
        unique_horses.to_csv(output_file, index=False, encoding='utf_8_sig')
        
        print(f"完了しました！")
        print(f"抽出された馬の総数: {len(unique_horses)} 頭")
        print(f"保存ファイル名: {output_file}")
        
    except FileNotFoundError:
        print("エラー: 'master_data.csv' が見つかりません。ファイル名を確認してください。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    extract_unique_horses()