import os
import re
import sys
import time
import urllib.parse
import pandas as pd
import requests

# 【設定】ファイル名の指定（★ご要望通り、さっきと違う名前に変更いたしましたわ！）
INPUT_HORSE_LIST = "unique_horse_list.csv"
OUTPUT_RAW_62 = "horses_raw_database_62.csv"  # ➔ 新しいCSVファイル名です
FAILED_LOG_FILE = "failed_horses.txt"  # 失敗した馬を記録するテキストファイル

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://db.netkeiba.com/",
}


def log_failure(horse_name, reason):
    """失敗した馬の名前と原因をテキストファイルに自動で追記するガード行動"""
    with open(FAILED_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"【失敗】 {horse_name} ➔ 原因: {reason}\n")


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
            print(
                f"  [!] HTTPエラー: ステータスコードが {response.status_code} です"
            )
            return "HTTP_ERROR"
        response.encoding = "euc-jp"

        tables = pd.read_html(response.text)
        pedigree_df = None
        for df in tables:
            if df.shape[1] >= 5:
                pedigree_df = df
                break

        if pedigree_df is None:
            return "NO_PEDIGREE_TABLE"

        flattened_clean_list = []
        for n in range(32):
            for col_idx in range(5):
                cell_value = str(pedigree_df.iat[n, col_idx]).strip()
                cell_value = re.sub(r"\s+", " ", cell_value)
                if cell_value == "nan":
                    cell_value = ""

                generation = col_idx + 1

                if generation == 5:
                    flattened_clean_list.append(cell_value)
                elif generation == 1:
                    if n == 0 or n == 16:
                        flattened_clean_list.append(cell_value)
                    else:
                        flattened_clean_list.append("")
                elif generation == 2:
                    if n % 8 == 0:
                        flattened_clean_list.append(cell_value)
                    else:
                        flattened_clean_list.append("")
                elif generation == 3:
                    if n % 4 == 0:
                        flattened_clean_list.append(cell_value)
                    else:
                        flattened_clean_list.append("")
                elif generation == 4:
                    if n % 2 == 0:
                        flattened_clean_list.append(cell_value)
                    else:
                        flattened_clean_list.append("")

        final_short_list = [val for val in flattened_clean_list if val != ""]

        row_data = {"target_horse_name": horse_name, "horse_id": horse_id}
        for idx, val in enumerate(final_short_list):
            row_data[f"blood_{idx+1}"] = val

        return row_data

    except Exception as e:
        print(f"  [!] 解析中にエラーが発生しました: {e}")
        return "PARSE_EXCEPTION"


def main():
    print("==================================================")
    print(" 【新ファイル名版】62列ありのまま一括ハントシステム")
    print("==================================================\n")

    if not os.path.exists(INPUT_HORSE_LIST):
        print(f"【エラー】リストファイル 『{INPUT_HORSE_LIST}』 がありません。")
        return

    horse_df = pd.read_csv(INPUT_HORSE_LIST, encoding="utf-8-sig")
    target_horses = horse_df["馬名"].dropna().tolist()
    total_horses = len(target_horses)

    print(
        f"➔ 準備完了: 『{INPUT_HORSE_LIST}』 から 【{total_horses}頭】 の馬名を正常に読み込みました。\n"
    )

    session = requests.Session()
    try:
        session.get("https://db.netkeiba.com/", headers=HEADERS, timeout=5)
        print("➔ ネット競馬への接続成功！自動ハントを開始いたしますわ。\n")
    except:
        print("【致命的エラー】ネット競馬への接続に失敗しました。")
        return

    print("-" * 60)
    print(" 進捗状況  |  馬名  |  ステータス詳細")
    print("-" * 60)

    for i, horse_name in enumerate(target_horses):
        horse_name = str(horse_name).strip()
        progress = f"[{i+1}/{total_horses}] ({(i+1)/total_horses*100:.1f}%)"

        print(f"{progress} 「{horse_name}」 の処理を開始...")

        # 1. 馬IDの特定に失敗した場合
        horse_id = get_horse_id(session, horse_name)
        if not horse_id:
            msg = "ネット競馬内で馬IDが見つかりませんでした（同名不在、または通信遮断）"
            print(f"  ❌ [失敗スキップ] {msg}")
            log_failure(horse_name, msg)
            print("-" * 40)
            continue

        # 2. 血統解析に失敗した場合
        horse_row_data = fetch_and_clean_pedigree_62(
            session, horse_id, horse_name
        )

        if (
            horse_row_data == "HTTP_ERROR"
            or horse_row_data == "NO_PEDIGREE_TABLE"
            or horse_row_data == "PARSE_EXCEPTION"
        ):
            msg = f"血統ページのデータ取得・解析に失敗しました（エラータイプ: {horse_row_data}）"
            print(f"  ❌ [失敗スキップ] {msg}")
            log_failure(horse_name, msg)
            print("-" * 40)
            continue
        elif horse_row_data is None:
            msg = "予期せぬ理由で血統データが取得できませんでした"
            print(f"  ❌ [失敗スキップ] {msg}")
            log_failure(horse_name, msg)
            print("-" * 40)
            continue

        # 3. 保存成功（★新しいファイル名に対して追記を行います）
        df_single = pd.DataFrame([horse_row_data])
        file_exists = os.path.exists(OUTPUT_RAW_62)
        df_single.to_csv(
            OUTPUT_RAW_62,
            mode="a",
            index=False,
            header=not file_exists,
            encoding="utf-8-sig",
        )

        print(
            f"  ★ [無事保存完了] 「{horse_name}」 のデータは 『{OUTPUT_RAW_62}』 の末尾に追記されました！"
        )
        print("-" * 40)

        time.sleep(2)

    print("\n" + "=" * 60)
    print(f" 【全ミッションが終了いたしました！】")
    print(f" 正常保存ファイル ➔ 『{OUTPUT_RAW_62}』")
    print(f" 失敗リストファイル ➔ 『{FAILED_LOG_FILE}』")
    print("=" * 60)


if __name__ == "__main__":
    main()