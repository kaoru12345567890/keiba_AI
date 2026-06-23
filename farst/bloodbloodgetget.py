import os
import time
import random
import pandas as pd
import requests
import threading
import urllib.parse
import re
import warnings

# 警告抑制
warnings.simplefilter('ignore', FutureWarning)

# 【設定】
INPUT_HORSE_LIST = "unique_horse_list.csv"
OUTPUT_RAW_62 = "filtered_horses_database.csv"
FAILED_LOG_FILE = "failed_horses.txt"
ERROR_THRESHOLD = 20
consecutive_errors = 0

# 【制御用】
write_lock = threading.Lock()
error_lock = threading.Lock()
thread_local = threading.local()

# UAリスト
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
]

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        idx = random.randint(0, len(USER_AGENTS) - 1)
        thread_local.ua_index = idx + 1 
        
        thread_local.session.headers.update({
            "User-Agent": USER_AGENTS[idx],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "ja-JP,ja;q=0.9",
            "Referer": "https://db.netkeiba.com/",
            "Upgrade-Insecure-Requests": "1"
        })
    return thread_local.session

def log_failure(horse_name, reason, ua_idx):
    with write_lock:
        with open(FAILED_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"【失敗】 {horse_name} ➔ 原因: {reason} (UA:{ua_idx})\n")
    print(f" ❌ [失敗] {horse_name} (原因: {reason}, UA:{ua_idx})")

def fetch_and_clean_pedigree(session, horse_id, horse_name):
    target_url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"
    try:
        time.sleep(random.uniform(5.0, 15.0))
        response = session.get(target_url, timeout=20)
        
        if response.status_code == 403: return "HTTP_403"
        if response.status_code != 200: return f"HTTP_{response.status_code}"
        
        response.encoding = "euc-jp"
        tables = pd.read_html(response.text)
        pedigree_df = next((df for df in tables if df.shape[1] >= 5), None)
        if pedigree_df is None: return "NO_TABLE"
        
        flat = []
        for n in range(32):
            for col in range(5):
                val = str(pedigree_df.iat[n, col]).strip()
                val = re.sub(r"\s+", " ", val)
                val = "" if val == "nan" else val
                if col == 4 or (col == 0 and n in [0, 16]) or (col == 1 and n % 8 == 0) or (col == 2 and n % 4 == 0) or (col == 3 and n % 2 == 0):
                    flat.append(val)
        
        row = {"target_horse_name": horse_name, "horse_id": horse_id}
        for i, val in enumerate([v for v in flat if v != ""][:62]):
            row[f"blood_{i+1}"] = val
        return row
    except Exception as e: return f"PARSE_{str(e)}"

def process_horse(horse_name):
    global consecutive_errors
    session = get_session()
    ua_idx = thread_local.ua_index
    encoded_name = urllib.parse.quote(horse_name.encode("euc-jp"))
    url = f"https://db.netkeiba.com/index.php?pid=horse_list&word={encoded_name}"
    
    try:
        res = session.get(url, timeout=15)
        if res.status_code != 200:
            raise Exception(f"HTTP_{res.status_code}")
            
        match = re.search(r"horse/(\d+)", res.url)
        if not match:
            raise Exception("ID_NOT_FOUND")
        
        horse_id = match.group(1)
        result = fetch_and_clean_pedigree(session, horse_id, horse_name)
        
        if isinstance(result, str):
            raise Exception(result)
        
        with error_lock:
            consecutive_errors = 0
        
        with write_lock:
            file_exists = os.path.exists(OUTPUT_RAW_62)
            pd.DataFrame([result]).to_csv(OUTPUT_RAW_62, mode="a", index=False, header=not file_exists, encoding="utf-8-sig")
            print(f" ✅ [成功] {horse_name} (UA:{ua_idx})")
            
    except Exception as e:
        with error_lock:
            consecutive_errors += 1
            current_errors = consecutive_errors
        
        log_failure(horse_name, str(e), ua_idx)
        
        if current_errors >= ERROR_THRESHOLD:
            print(f"\n 🚨 警告: 連続エラーが {ERROR_THRESHOLD} 回に達しました。停止します。")
            os._exit(1)

def main():
    if not os.path.exists(INPUT_HORSE_LIST):
        print("入力ファイルが見つかりません。")
        return

    df = pd.read_csv(INPUT_HORSE_LIST, encoding="utf-8-sig")
    target_horses = df["馬名"].dropna().unique().tolist()
    
    if os.path.exists(OUTPUT_RAW_62):
        finished = pd.read_csv(OUTPUT_RAW_62, encoding="utf-8-sig")["target_horse_name"].unique().tolist()
        target_horses = [n for n in target_horses if n not in finished]

    print(f"--- ハント開始: 残り {len(target_horses)} 頭 ---")
    random.shuffle(target_horses)
    
    i = 0
    batch_count = 1
    
    while i < len(target_horses):
        # 毎回 50〜100 の間でランダムなバッチサイズを決定
        current_batch_size = random.randint(70, 130)
        
        # 今回のバッチを取り出す
        batch = target_horses[i : i + current_batch_size]
        remaining = len(target_horses) - i
        
        print(f"--- {batch_count} バッチ目開始 (残り {remaining} 頭 / 今回 {len(batch)} 頭) ---")
        
        if hasattr(thread_local, "session"):
            delattr(thread_local, "session")
            
        for horse in batch:
            process_horse(horse)
            
        i += current_batch_size
        batch_count += 1
        
        # 次のバッチがある場合のみ休憩
        if i < len(target_horses):
            rest_time = random.uniform(30.0, 60.0)
            print(f"➔ バッチ終了。休憩中 ({rest_time:.1f}秒)...")
            time.sleep(rest_time)

if __name__ == "__main__":
    main()