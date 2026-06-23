import pandas as pd
import numpy as np
from tqdm import tqdm

# 1. データの読み込み
df = pd.read_csv('processed_9_data.csv', header=None)
pd.set_option('future.no_silent_downcasting', True)

def to_numeric_safe(series):
    return pd.to_numeric(series.astype(str).str.extract('(\d+)')[0], errors='coerce').fillna(0)

# 2. 並び替え
df = df.sort_values(by=[1, 2, 3, 5, 8], ascending=True)

# 3. グループごとの時系列処理
def calc_past_stats(group):
    res = group.copy()
    
    col17_num = to_numeric_safe(group[17])
    col39_num = to_numeric_safe(group[39])
    
    prev_col17 = col17_num.shift(1)
    prev_col39 = col39_num.shift(1)
    
    res['過去出走回数'] = prev_col17.rolling(window=5, min_periods=1).count().fillna(0)
    res['平均着順'] = prev_col17.rolling(window=5, min_periods=1).mean()
    res['連対率'] = prev_col17.isin([1, 2]).rolling(window=5, min_periods=1).mean()
    res['複勝率'] = prev_col17.isin([1, 2, 3]).rolling(window=5, min_periods=1).mean()
    res['平均上がり偏差値'] = prev_col39.rolling(window=5, min_periods=1).mean()
    
    return res.fillna(0)

# 4. プログレスバー付きでグループ化して適用
# グループ化されたオブジェクトをリスト化してから tqdm でラップすることで進捗を表示します
grouped = df.groupby(20, group_keys=False)
tqdm.pandas(desc="集計処理中")
df_final = grouped.progress_apply(calc_past_stats)

# 5. 保存
df_final.to_csv('processed_10_data.csv', index=False, header=False)

print("完了: 処理が終了しました。processed_10_data.csv に保存しました。")