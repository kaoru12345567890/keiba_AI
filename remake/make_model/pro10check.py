import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 1. データの読み込み
df = pd.read_csv('processed_10_data.csv', header=None, low_memory=False)
col_names = [
    'レースID', '年', '月', '日', '曜日', '場所', '回', '日目', 'レース目', 'レース名', 
    '天気', '馬場状態', 'レース条件', '芝ダート', '距離', '回り', '出走数', '着順', 
    '枠番', '馬番', '馬名', '性別年齢', '斤量', '騎手', 'タイム', '着差', 'ペース', 
    '通過順', '上り3ハロン', '単勝', '人気', '馬体重', '体重増減', '所属', '調教師名', '馬主', 
    '賞金', '脚質スコア', '脚質ラベル', '上がり偏差値', '性別', '年齢', '過去出走回数', 
    '過去平均着順', '過去連対率', '過去複勝率', '過去平均上がり偏差値'
]
df.columns = col_names

# 2. エンコード対象の列定義
le_cols = ['天気', '馬場状態', 'レース条件', '芝ダート', '回り', '騎手', '調教師名', '所属', '性別', '曜日']

# 3. エンコード実行と対応表の作成
# 対応表を保存するための辞書
mapping_dict = {}

for col in le_cols:
    le = LabelEncoder()
    # 文字列として変換
    df[col] = le.fit_transform(df[col].astype(str))
    
    # 対応表を保存
    mapping_dict[col] = pd.DataFrame({'Original': le.classes_, 'Encoded': le.transform(le.classes_)})
    
    # 変換ルールをファイルとして保存
    joblib.dump(le, f'le_{col}.pkl')

# 4. 不要な列の削除（学習に使わないデータやエンコードしていない文字列データ）
keep_cols = [c for c in df.columns if c in le_cols or c not in [
    'レースID', 'レース名', '馬名', '馬主', 'タイム', '着差', '通過順', '賞金', '性別年齢', '上り3ハロン', '単勝', '人気', '上がり偏差値'
]]
model_df = df[keep_cols]

# 5. ファイル出力
# 学習用データ
model_df.to_csv('model_data.csv', index=False, encoding='utf_8_sig')

# 対応表を2つのファイルに出力（列数が多いので分割）
half = len(le_cols) // 2
pd.concat([mapping_dict[col].assign(Column=col) for col in le_cols[:half]]).to_csv('mapping_1.csv', index=False, encoding='utf_8_sig')
pd.concat([mapping_dict[col].assign(Column=col) for col in le_cols[half:]]).to_csv('mapping_2.csv', index=False, encoding='utf_8_sig')

print("--- 完了 ---")
print("1. model_data.csv を作成しました")
print("2. mapping_1.csv と mapping_2.csv に対応表を出力しました")