天気 = "\u6674" #@param ["晴", "曇", "雨", "小雨", "雪"]
馬場状態 = "\u826F" #@param ["良", "稍重", "重", "不良"]

import pandas as pd
import time
import random
import re
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import requests
import datetime

# 開催情報の入力
raw_input = input("開催情報を入力してください (例: 202606130101): ")
if len(raw_input) != 12:
    print("エラー: 入力形式が違います。12桁の数字で入力してください。")
    raw_input = input("開催情報を入力してください (例: 202606130101): ")
    exit()
year, month, day = int(raw_input[0:4]), int(raw_input[4:6]), int(raw_input[6:8])
kai, nichime = int(raw_input[8:10]), int(raw_input[10:12])

output_filename = f"hakodate_{year}{month:02d}{day:02d}.csv"

date_obj = datetime.date(year, month, day)
weekday_list = ["月", "火", "水", "木", "金", "土", "日"]
day_of_week = weekday_list[date_obj.weekday()]

# 場所コードは函館の 02 で固定
place_val = 2

# 入力内容の確認
print("-" * 30)
print(f"日付: {year}年{month}月{day}日")
print(f"開催場所: 函館 (コード: {place_val:02d})")
print(f"開催回次: 第{int(kai)}回 {int(nichime)}日目")
print(f"曜日: {day_of_week}")
print("-" * 30)

# 騎手・調教師のフルネームを取得するための関数
def fetch_real_fullname_playwright(page, url, cache_dict):
    if not url or url in cache_dict: 
        return cache_dict.get(url, "")
    
    # 新しいタブで詳細ページを開く
    new_page = page.context.new_page()
    try:
        new_page.goto(url, wait_until="domcontentloaded", timeout=10000)
        # Netkeibaの詳細ページのh1を取得
        h1 = new_page.query_selector("#db_main_box h1")
        if h1:
            name = re.sub(r"\(.*\)|（.*）", "", h1.inner_text().strip())
            cache_dict[url] = name
            new_page.close()
            return name
    except Exception as e:
        print(f"名前取得エラー: {e}")
    
    new_page.close()
    return ""

def get_race_data():
    all_race_data = []
    jockey_cache, trainer_cache = {}, {}
    session = requests.Session()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        page = context.new_page()

        for race_num in range(1, 13):
            race_id = f"{year}02{kai:02d}{nichime:02d}{race_num:02d}"
            url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
            print(f"解析中: {race_id} ({race_num}R)")
            
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                
                # レース基本情報
                # レース名の要素を取得
                race_name = page.query_selector(".RaceName")
                # 要素の中身（文字列）を取得
                recename = race_name.inner_text().strip() if race_name else "不明"

                # 重賞なら名前を更新
                if race_name and race_name.query_selector("span.Icon_GradeType1, span.Icon_GradeType2, span.Icon_GradeType3"):
                    recename = f"第0回{recename}"
                
                rows = page.query_selector_all("tr.HorseList")
                for row in rows:
                    cols = row.query_selector_all("td")
                    if len(cols) < 8: continue

                    # レース情報の解析（コース・距離・回り）
                    race_data_div = page.query_selector(".RaceData01")
                    race_info_text = race_data_div.inner_text().strip() if race_data_div else ""

                    track_type, direction, distance = "芝", "右", 1200
                    if race_info_text:
                        track_type = "ダ" if "ダ" in race_info_text else "芝"
                        direction = "左" if "左" in race_info_text else ("直" if "直" in race_info_text else "右")
                        dist_match = re.search(r"(\d+)m", race_info_text)
                        if dist_match: distance = int(dist_match.group(1))

                    # 2. レース条件
                    race_data_02 = page.query_selector(".RaceData02")
                    race_condition = "(混)[指]馬齢"
                    if race_data_02:
                        raw_text = race_data_02.inner_text().strip()
                        c_clean = re.sub(r"\d+回[^\d]+\d+日目|(新馬|未勝利|１勝クラス|1勝クラス|２勝クラス|2勝クラス|３勝クラス|3勝クラス|オープン|OP|重賞|GI|G1|GII|G2|GIII|G3|サラ系|アラブ系)|\d+歳(以上|未満)?|\d+頭|本賞金.*$", "", raw_text)
                        race_condition = re.sub(r"([\]\)])\s+", r"\1", " ".join(c_clean.split()).strip())
                        if not race_condition: race_condition = "定量"

                    # --- あなたの解析ロジックを統合 ---
                    waku = cols[0].inner_text().strip()
                    umaban = cols[1].inner_text().strip()
                    horse_name = cols[3].inner_text().strip()
                    seirei = cols[4].inner_text().strip()
                    seibetsu, age = (seirei[0], seirei[1:]) if len(seirei) > 1 else ("", seirei)
                    weight = cols[5].inner_text().strip()
                    
                    # 騎手・調教師の抽出とクリーニング処理
                    jockey_td = cols[6]
                    jockey_a = jockey_td.query_selector("a")
                    j_url = jockey_a.get_attribute("href") if jockey_a else ""

                    # 関数を呼び出し
                    j_full = fetch_real_fullname_playwright(page, j_url, jockey_cache)

                    # 騎手名クリーニング：記号削除 + 空白を詰める
                    raw_jockey = jockey_td.inner_text()
                    # 1. 記号削除、2. すべての空白(\s+)を削除、3. 前後の空白を削除
                    jockey = j_full if j_full else re.sub(r"[☆▲△▼◇★▶縲咲┯\s ]", "", raw_jockey).strip()

                    trainer_td = cols[7]
                    trainer_a = trainer_td.query_selector("a")
                    t_url = trainer_a.get_attribute("href") if trainer_a else ""
                    t_full = fetch_real_fullname_playwright(page, t_url, trainer_cache)

                    # 調教師名クリーニング：所属部分(2文字)を除去した後に空白処理
                    raw_trainer = trainer_td.inner_text()
                    belongs = "東" if "美浦" in raw_trainer else ("西" if "栗東" in raw_trainer else "地")
                    clean_trainer_name = re.sub(r"[美浦栗東地\s  ]", "", raw_trainer[2:])
                    trainer = t_full if t_full else clean_trainer_name

                    # 馬体重分解ロジック
                    batai_td = cols[8].inner_text().strip()
                    batai_match = re.search(r"(\d+)\((.*?)\)", batai_td)
                    batai_val, zougen_val = (int(batai_match.group(1)), int(batai_match.group(2))) if batai_match else (0, 0)
                    
                    # 単勝
                    odds = cols[9].inner_text().strip()
                    
                    # データ格納
                    all_race_data.append({
                        "レースID": race_id, "年": int(year), "月": month, "日": day, "曜日": day_of_week, "場所": place_val, "回": int(kai), "日目": int(nichime), "レース目": race_num, 
                        "レース名": recename, "天気": 天気, "馬場状態": 馬場状態, "レース条件": race_condition, "芝ダート": track_type, "距離": distance, "回り": direction, 
                        "出走数": len(rows)-2, "枠番": waku, "馬番": umaban, "馬名": horse_name, "性別": seibetsu, "年齢": int(age) if age.isdigit() else age, "斤量": weight,
                        "騎手": re.sub(r"\s+", " ", jockey), "所属": belongs, "調教師名": re.sub(r"\s+", " ", trainer), "馬体重": batai_val, "体重増減": zougen_val,
                        "過去出走回数": 0, "過去平均着順": 0, "過去連対率": 0, "過去複勝率": 0, "過去平均上がり偏差値": 0, "単勝": odds,
                    })
                
                time.sleep(random.uniform(2.0, 4.0))
            except Exception as e:
                print(f"ID:{race_id}でエラー: {e}")
        browser.close()
    
    return pd.DataFrame(all_race_data)

# 実行
df = get_race_data()
df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"完了: {output_filename}")