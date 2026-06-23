import pandas as pd
import re
import os

def process_racing_data(input_path, output_path):
    # 1. データの読み込み
    if not os.path.exists(input_path):
        print(f"エラー: {input_path} が見つかりません。")
        return

    # ヘッダーありと仮定して読み込み（ヘッダーがない場合はheader=Noneにしてください）
    df = pd.read_csv(input_path)
    
    # 2. 馬体重の分解 (元のデータの34番目: '馬体重増減')
    # 480(+4) のようなデータを 分解する
    def split_weight(val):
        val_str = str(val)
        if '(' not in val_str:
            return pd.Series([val, 0])
        # 正規表現: 数字(\d+)とカッコの中身(.*)を抽出
        match = re.match(r'(\d+)\((.*)\)', val_str)
        if match:
            return pd.Series([float(match.group(1)), float(match.group(2))])
        return pd.Series([val, 0])

    weight_split = df['馬体重増減'].apply(split_weight)
    df['馬体重'] = weight_split[0]
    df['体重増減'] = weight_split[1]
    
    # 3. 調教師情報の分解 (元のデータの38番目: '調教師')
    # [西]松永幹夫 のようなデータを 分解する
    def split_trainer(val):
        val_str = str(val)
        if '[' not in val_str:
            return pd.Series(['不明', val])
        # [東/西]を抽出して分ける
        match = re.match(r'\[(.*)\](.*)', val_str)
        if match:
            return pd.Series([match.group(1), match.group(2)])
        return pd.Series(['不明', val])

    trainer_split = df['調教師'].apply(split_trainer)
    df['所属'] = trainer_split[0]
    df['調教師名'] = trainer_split[1]
    
    # 4. 新しいCSVとして保存
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"成功: {output_path} が作成されました！")

# --- メイン実行部分 ---
if __name__ == "__main__":
    # ここに読み込むファイル名と、新しく作るファイル名を指定
    input_csv = 'master_data.csv'  # お持ちの元データ名
    output_csv = 'processed_1_data.csv' # 新しく作るファイル名
    
    process_racing_data(input_csv, output_csv)