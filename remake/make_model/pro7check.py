import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

# 1. データの読み込み
df = pd.read_csv('processed_7_data.csv', header=None)

# 2. 性別年齢(21番)を分割
def split_sex_age(row):
    val = str(row[21])
    sex = re.sub(r'\d+', '', val)
    age = re.sub(r'\D+', '', val)
    return sex, int(age) if age else 0

# 一旦分割結果を格納（数値化前）
temp_results = df.apply(split_sex_age, axis=1)
df[40], df[41] = zip(*temp_results)

# 3. 分類結果の確認と数値化
# どの文字が何番になったかを表示するための処理
le_sex = LabelEncoder()
encoded_sex = le_sex.fit_transform(df[40].astype(str))

# 各種類の件数をカウント
df[40] = encoded_sex
count_data = df[40].value_counts()

print("="*30)
print("【性別の分類詳細】")
print("対応表 (ID: 文字):")
for i, name in enumerate(le_sex.classes_):
    print(f" ID {i}: {name}")

print("\nIDごとのデータ件数:")
print(count_data)
print("="*30)

# 4. 保存
df.to_csv('processed_8_data.csv', index=False, header=False)
print("保存完了: processed_8_data.csv")