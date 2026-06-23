import pandas as pd
import os

def check_integrity(original_path, processed_path):
    # ファイルの存在確認
    if not os.path.exists(original_path) or not os.path.exists(processed_path):
        print("エラー: 指定されたファイルが見つかりません。")
        return

    # 読み込み
    df_orig = pd.read_csv(original_path, low_memory=False)
    df_proc = pd.read_csv(processed_path, low_memory=False)

    print("--- 整合性チェック結果 ---")
    
    # 1. 行数のチェック
    orig_rows = len(df_orig)
    proc_rows = len(df_proc)
    print(f"行数: {orig_rows} (元) -> {proc_rows} (加工後)")
    if orig_rows == proc_rows:
        print("✅ 行数は一致しています。")
    else:
        print("❌ 行数が一致しません！")

    # 2. 列数のチェック（+4列されているか）
    orig_cols = len(df_orig.columns)
    proc_cols = len(df_proc.columns)
    print(f"列数: {orig_cols} (元) -> {proc_cols} (加工後)")
    if proc_cols == orig_cols + 4:
        print("✅ 列数は期待通り（+4列）増加しています。")
    else:
        print(f"❌ 列数が期待通りではありません。")

    # 3. 欠損値（ヌル）のチェック
    # 新しく作ったはずの列にデータが入っているか確認
    target_cols = ['馬体重', '体重増減', '所属', '調教師名']
    # 存在チェック
    missing_cols = [c for c in target_cols if c not in df_proc.columns]
    
    if not missing_cols:
        null_counts = df_proc[target_cols].isnull().sum()
        print("\n--- 新規列の欠損値チェック ---")
        print(null_counts)
        if null_counts.sum() == 0:
            print("✅ 全ての新規列にデータが埋まっています。")
        else:
            print("⚠️ 一部の行でデータが空（欠損）です。")
    else:
        print(f"❌ 新規列が見つかりません: {missing_cols}")

# --- 実行設定 ---
# ここを実際のファイル名に書き換えてください
original_file = 'master_data.csv'  # 元データ
processed_file = 'processed_1_data.csv' # 加工後のデータ

if __name__ == "__main__":
    check_integrity(original_file, processed_file)