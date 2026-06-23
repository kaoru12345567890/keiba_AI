import pandas as pd
import os

def reorder_and_save_data(input_path, output_path):
    df = pd.read_csv(input_path, low_memory=False)
    
    # 1. 理想の列順を定義（元の並びに加工済みデータを組み込む）
    # 元の構造: ... 着順, [馬体重増減], ..., [調教師], 馬主, 賞金
    # 新しい構造: ... 着順, [馬体重], [体重増減], ..., [所属], [調教師名], 馬主, 賞金
    
    new_order = [
        'レースID', '年', '月', '日', '曜日', '場所', '回', '日目', 'レース目', 
        'レース名', '天気', '馬場状態', 'レース条件', '芝ダート', '距離', '回り', 
        '出走馬数', '枠番', '馬番', '人気', '馬名', '性別年齢', '斤量', '騎手', 
        'タイム', '着差', 'ペース', '通過順', '上がり3F', '単勝', '着順', 
        '馬体重', '体重増減', '所属', '調教師名', '馬主', '賞金'
    ]
    
    # 2. 順序適用
    # もしデータ内に存在しない列があった場合の保険として、存在するものだけを抽出
    existing_cols = [c for c in new_order if c in df.columns]
    df = df[existing_cols]
    
    # 3. 保存
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"成功: {output_path} に保存しました。列の並び替えが完了しました。")

if __name__ == "__main__":
    reorder_and_save_data('processed_2_data.csv', 'processed_3_data.csv')