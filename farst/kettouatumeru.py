import os
import re
import sys
import time
import urllib.parse
import pandas as pd
import requests

TARGET_HORSES = ["アーモンドアイ", "コントレイル"]
OUTPUT_FILE = "horses_perfect_short_62.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://db.netkeiba.com/",
}


def get_horse_id(session, horse_name):
    encoded_name = urllib.parse.quote(horse_name.encode("euc-jp"))
    search_url = (
        f"https://db.netkeiba.com/index.php?pid=horse_list&word={encoded_name}"
    )
    try:
        response = session.get(search_url, headers=HEADERS, timeout=10)
        response.encoding = "euc-jp"
        if "horse/" in response.url:
            return re.search(r"horse/(\d+)", response.url).group(1)
        else:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "html.parser")
            all_links = soup.find_all("a", href=re.compile(r"/horse/\d+"))
            if not all_links:
                return None
            return re.search(r"horse/(\d+)", all_links[0].get("href")).group(1)
    except:
        return None


def fetch_and_clean_pedigree_62(session, horse_id, horse_name):
    target_url = f"https://db.netkeiba.com/horse/ped/{horse_id}/"

    try:
        response = session.get(target_url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None
        response.encoding = "euc-jp"

        tables = pd.read_html(response.text)
        pedigree_df = None
        for df in tables:
            if df.shape[1] >= 5:
                pedigree_df = df
                break

        if pedigree_df is None:
            return None

        flattened_clean_list = []
        for n in range(32):
            for col_idx in range(5):
                cell_value = str(pedigree_df.iat[n, col_idx]).strip()
                cell_value = re.sub(r"\s+", " ", cell_value)
                if cell_value == "nan":
                    cell_value = ""

                generation = col_idx + 1

                # 5代前は無条件で100%すべて残す
                if generation == 5:
                    flattened_clean_list.append(cell_value)

                # ★修正ポイント：1代前は n=0（パパ）だけでなく、n=16（ママ）の特等席も絶対に守る！
                elif generation == 1:
                    if n == 0 or n == 16:
                        flattened_clean_list.append(cell_value)
                    else:
                        flattened_clean_list.append("")

                # 2代前：8の倍数の行（0, 8, 16, 24）だけを残す
                elif generation == 2:
                    if n % 8 == 0:
                        flattened_clean_list.append(cell_value)
                    else:
                        flattened_clean_list.append("")

                # 3代前：4の倍数の行（0, 4, 8, 12...）だけを残す
                elif generation == 3:
                    if n % 4 == 0:
                        flattened_clean_list.append(cell_value)
                    else:
                        flattened_clean_list.append("")

                # 4代前：2の倍数の行（0, 2, 4, 6...）だけを残す
                elif generation == 4:
                    if n % 2 == 0:
                        flattened_clean_list.append(cell_value)
                    else:
                        flattened_clean_list.append("")

        # 処理が終わった最後の瞬間に、空欄を一斉に排除してギュッと詰める行動
        final_short_list = [val for val in flattened_clean_list if val != ""]

        # 辞書データにマッピング（今度はヘッダーも blood_1 〜 blood_62 までぴったり作られます）
        row_data = {"target_horse_name": horse_name, "horse_id": horse_id}
        for idx, val in enumerate(final_short_list):
            row_data[f"blood_{idx+1}"] = val

        return row_data

    except Exception as e:
        print(f"[{horse_name}] 解析エラー: {e}")
        return None


def main():
    print("==================================================")
    print(" 【完全修正版】1頭完結型・62列正確短縮追記システム")
    print("==================================================\n")

    session = requests.Session()
    try:
        session.get("https://db.netkeiba.com/", headers=HEADERS, timeout=5)
    except:
        return

    for i, horse_name in enumerate(TARGET_HORSES):
        print(f"[{i+1}/{len(TARGET_HORSES)}] 「{horse_name}」をハント中...")

        horse_id = get_horse_id(session, horse_name)
        if not horse_id:
            continue

        horse_row_data = fetch_and_clean_pedigree_62(
            session, horse_id, horse_name
        )
        if horse_row_data is None:
            continue

        df_single = pd.DataFrame([horse_row_data])

        file_exists = os.path.exists(OUTPUT_FILE)
        df_single.to_csv(
            OUTPUT_FILE,
            mode="a",
            index=False,
            header=not file_exists,
            encoding="utf-8-sig",
        )
        print(f" ➔ [成功] ママのデータも守って62列で追記保存しました。")
        time.sleep(1)


if __name__ == "__main__":
    main()