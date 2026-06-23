import pandas as pd
import os

def check_integrity_final(processed1_path, processed2_path):
    # ファイルの存在確認
    if not os.path.exists(processed1_path) or not os.path.exists(processed2_path):
        print("エラー: 指定されたファイルが見つかりません。")
        return

    # 読み込み
    df_proc1 = pd.read_csv(processed1_path, low_memory=False)
    df_proc2 = pd.read_csv(processed2_path, low_memory=False)

    print("--- 最終整合性チェック結果 ---")
    
    # 1. 行数のチェック（行数は変わらないはず）
    print(f"行数: {len(df_proc1)} (加工前1) -> {len(df_proc2)} (加工後2)")
    if len(df_proc1) == len(df_proc2):
        print("✅ 行数は維持されています。")
    else:
        print("❌ 行数が変わっています！")

    # 2. 不要な列が消えたかチェック
    # '?'を含む列や、加工前の列が残っていないか確認
    drop_targets = ['馬体重増減', '調教師']
    # '?'を含む列を抽出
    q_cols = [c for c in df_proc2.columns if '?' in c]
    
    leftover_cols = [c for c in drop_targets if c in df_proc2.columns]
    
    if len(q_cols) == 0 and len(leftover_cols) == 0:
        print("✅ 不要な列（はてな系・元データ）はすべて削除されました。")
    else:
        print(f"❌ まだ不要な列が残っています: {q_cols + leftover_cols}")

    # 3. 必要な列が存在するかチェック
    required_cols = ['馬体重', '体重増減', '所属', '調教師名']
    missing_req = [c for c in required_cols if c not in df_proc2.columns]
    
    if not missing_req:
        print("✅ 必要な加工済みデータはすべて揃っています。")
        # 欠損値チェック
        null_counts = df_proc2[required_cols].isnull().sum()
        print(f"\n--- 新規列の欠損値チェック ---\n{null_counts}")
    else:
        print(f"❌ 必要な列が見つかりません: {missing_req}")

# --- 実行設定 ---
if __name__ == "__main__":
    # 読み込み元のファイル名と、新しく作ったファイル名を指定
    file1 = 'processed_1_data.csv'
    file2 = 'processed_2_data.csv'
    
    check_integrity_final(file1, file2)