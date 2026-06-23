import pandas as pd

def check_csv_structure():
    file_path = r"C:\keiba_AI\remake\hakodate_2026020102.csv"
    
    # 1. CSV読み込み
    df = pd.read_csv(file_path)
    
    # 列名の表示
    print(f"列名: {df.columns.tolist()}")
    
    # 2. 1Rの1番データがあるか確認
    sample_race = 1
    sample_umaban = 1
    
    mask = (df['レース目'] == sample_race) & (df['馬番'] == sample_umaban)
    
    if mask.any():
        print(f"成功: {sample_race}R {sample_umaban}番 の行は見つかりました。")
        print(df.loc[mask, ['レース目', '馬番', '馬体重']])
    else:
        print(f"失敗: {sample_race}R {sample_umaban}番 の行が見つかりません。")
        print("CSV内の「レース目」や「馬番」が数値であるか確認してください。")
        
        # データの型を表示
        print(f"列の型:\n{df.dtypes}")

if __name__ == "__main__":
    check_csv_structure()