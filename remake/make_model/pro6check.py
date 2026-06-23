import pandas as pd

# ファイルパス設定
input_file = 'processed_6_data.csv'
output_file = 'processed_7_data.csv'
error_file = 'error_races.txt'

# インデックス定義
IDX_RACE_ID = 0
IDX_PACE = 26  # ペース指数が含まれる列

# 1. データの読み込み（最初の2行を確実にスキップ）
print("データを読み込み中...")
df = pd.read_csv(input_file, skiprows=2, header=None, low_memory=False)

# 2. ペース情報の補完とエラー抽出
print("ペース情報の適用とエラーチェック中...")

# 数値変換を試みる（エラーはNaNにする）
df[IDX_PACE] = pd.to_numeric(df[IDX_PACE], errors='coerce')

# ペース情報が1つも存在しないレースIDを特定
grouped = df.groupby(IDX_RACE_ID)[IDX_PACE]
missing_mask = grouped.transform(lambda x: x.isna().all())
error_races = df[missing_mask][IDX_RACE_ID].unique()

# エラーレースをテキストファイルに書き出し
with open(error_file, 'w') as f:
    for race_id in error_races:
        f.write(f"{race_id}\n")

# レースID単位で値をコピーして全馬にペース情報を反映
# bfill()とffill()で、レース内のどこかに値があれば全体に行き渡らせる
df[IDX_PACE] = df.groupby(IDX_RACE_ID)[IDX_PACE].transform(lambda x: x.bfill().ffill())

# それでも埋まらない場合は全体の平均で補完
df[IDX_PACE] = df[IDX_PACE].fillna(df[IDX_PACE].mean())

# 3. データの保存
# indexもheaderも出力しないことで、純粋なデータだけのCSVを作成
df.to_csv(output_file, index=False, header=False)

print("-" * 30)
print(f"処理が完了しました！")
print(f"作成ファイル: {output_file}")
print(f"ペース欠損のあったレースIDを {error_file} に保存しました。")
print(f"全データ件数: {len(df)}件")
print("-" * 30)