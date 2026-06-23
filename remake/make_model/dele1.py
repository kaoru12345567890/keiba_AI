import pandas as pd
import re
import os

def clean_and_organize_data(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"エラー: {input_path} が見つかりません。")
        return
    
    df = pd.read_csv(input_path, low_memory=False)
    
    # 1. 加工ロジックの再確認（確実にきれいに分ける）
    # 馬体重の分解
    def split_weight(val):
        val_str = str(val)
        if '(' not in val_str: return pd.Series([val, 0])
        match = re.match(r'(\d+)\((.*)\)', val_str)
        if match: return pd.Series([float(match.group(1)), float(match.group(2))])
        return pd.Series([val, 0])

    # 調教師の分解
    def split_trainer(val):
        val_str = str(val)
        if '[' not in val_str: return pd.Series(['不明', val])
        match = re.match(r'\[(.*)\](.*)', val_str)
        if match: return pd.Series([match.group(1), match.group(2)])
        return pd.Series(['不明', val])

    # 既に加工済みの列があるか確認し、なければ再生成する
    if '馬体重' not in df.columns or '所属' not in df.columns:
        weight_split = df['馬体重増減'].apply(split_weight)
        df['馬体重'] = weight_split[0]
        df['体重増減'] = weight_split[1]
        
        trainer_split = df['調教師'].apply(split_trainer)
        df['所属'] = trainer_split[0]
        df['調教師名'] = trainer_split[1]

    # 2. 不要な列の徹底削除
    # ?系、加工前の元列、重複する可能性がある一時的な列をすべて指定
    cols_to_drop = [
        '馬体重増減', '調教師',  # 加工前の元データ
        '?', '?.1', '?.2', '?.3', '?.4', '?.5', '?.6', # はてな系
    ]
    
    # 実際に存在する列だけを削除対象にする
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 3. 最後に並び順を整える（見やすさと整理）
    # 必要な列を先頭に持ってきて、残りを後ろに配置
    essential_cols = ['レースID', '馬名', '馬体重', '体重増減', '所属', '調教師名']
    remaining_cols = [c for c in df.columns if c not in essential_cols]
    df = df[essential_cols + remaining_cols]

    # 4. 保存
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"成功: {output_path} に保存しました。")
    print(f"整理後の列数: {len(df.columns)}")

if __name__ == "__main__":
    clean_and_organize_data('processed_1_data.csv', 'processed_2_data.csv')