import pandas as pd

def full_verify_by_header(raw_path, proc_path, sample_n=10):
    df_raw = pd.read_csv(raw_path, low_memory=False)
    df_proc = pd.read_csv(proc_path, low_memory=False)

    # ランダムにインデックスを選択
    random_indices = df_raw.sample(n=sample_n).index

    print(f"--- 列名(ヘッダー)ベースの詳細比較 ({sample_n}件) ---")
    
    for i, idx in enumerate(random_indices):
        print(f"\n" + "="*80)
        print(f" [ランダム抽出 {i+1} 件目: インデックス {idx}]")
        print("="*80)
        
        row_raw = df_raw.iloc[idx]
        row_proc = df_proc.iloc[idx]
        
        # 全ての列名を取得して比較
        # 加工前の列と加工後の列をすべて列挙して表示します
        all_cols = set(df_raw.columns) | set(df_proc.columns)
        
        for col in sorted(list(all_cols)):
            val_raw = row_raw.get(col, "--- (なし) ---")
            val_proc = row_proc.get(col, "--- (なし) ---")
            
            # 値が同じか判定（型が違う場合を考慮して文字列化して比較）
            status = "OK" if str(val_raw) == str(val_proc) else "⚠️ 差異あり"
            
            print(f"{col:15} | 前: {str(val_raw):15} | 後: {str(val_proc):15} | {status}")

if __name__ == "__main__":
    full_verify_by_header('processed_1_data.csv', 'processed_3_data.csv', sample_n=10)