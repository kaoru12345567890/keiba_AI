import numpy as np
import pandas as pd
import statsmodels.api as sm

# データの準備
data = {
    "A": [38, 14, 100, 76, 69, 56, 36, 24, 24, 90, 84, 51, 45, 55, 69, 17, 80, 87, 49, 39],
    "B": [49, 74, 33, 24, 25, 55, 68, 25, 13, 33, 10, 29, 29, 15, 68, 31, 47, 16, 59, 68]
}

df = pd.DataFrame(data)

# 従属変数(y)と独立変数(x)の設定
y = df["A"]
X = df["B"]

# statsmodelsではデフォルトで定数項が含まれないため、明示的に追加する
X = sm.add_constant(X)

# 線形回帰モデルの構築と適合
model = sm.OLS(y, X)
results = model.fit()

# 分析結果のサマリーを表示
print(results.summary())