import pandas as pd

# ファイルパスの設定
input_file = 'processed_3_data.csv'

# インデックス定義（着順のインデックスを正確に指定）
IDX_CHAKUJUN = 17 

# データ読み込み
df = pd.read_csv(input_file, header=None, low_memory=False)

# 着順列を文字列型にして判定
# 列全体がobject型の場合もあるのでastype(str)で統一します
ser_chaku = df[IDX_CHAKUJUN].astype(str)

# それぞれの文字が含まれる行をカウント
count_jo = ser_chaku.str.contains('除', na=False).sum()
count_tor = ser_chaku.str.contains('取', na=False).sum()
count_chu = ser_chaku.str.contains('中', na=False).sum()

print(f"--- 着順列（インデックス {IDX_CHAKUJUN}）の異常データ ---")
print(f"『除』（除外）: {count_jo} 件")
print(f"『取』（取消）: {count_tor} 件")
print(f"『中』（中止）: {count_chu} 件")
print(f"合計異常件数: {count_jo + count_tor + count_chu} 件")