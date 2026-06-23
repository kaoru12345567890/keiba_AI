#@title 【設定】本日のレース環境入力フォーム { run: "auto" }
#@markdown ※右側のメニューを選択してからセルを実行してください。

天気 = "\u6674" #@param ["晴", "曇", "雨", "小雨", "雪"]
馬場状態 = "\u826F" #@param ["良", "稍重", "重", "不良"]

# ==============================================================================
# 以下、全自動スクレイピング＆過去データ合体コード
# ==============================================================================
import re
import time
import datetime
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# 開催情報を一括入力 (例: 202606130101)
# 構成: 年(4桁) + 月(2桁) + 日(2桁) + 回(2桁) + 日目(2桁)
raw_input = input("開催情報を入力してください (例: 202606130101): ")
if len(raw_input) != 12:
    print("エラー: 入力形式が違います。12桁の数字で入力してください。")
    raw_input = input("開催情報を入力してください (例: 202606130101): ")
    exit()

# 文字列から各情報を抽出
year = int(raw_input[0:4])
month = int(raw_input[4:6])
day = int(raw_input[6:8])
kai = int(raw_input[8:10])
nichime = int(raw_input[10:12])

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

# 日付入りファイル名を自動生成 (例: hakodate_20260613.csv)
output_filename = f"hakodate_{year}02{kai:02d}{nichime:02d}.csv"

all_race_data = []
jockey_cache = {}
trainer_cache = {}

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

def fetch_real_fullname(raw_url, cache_dict, is_jockey=True):
    if not raw_url or raw_url in cache_dict:
        return cache_dict.get(raw_url, "")
    id_match = re.search(r"(\d+)", raw_url)
    if not id_match: return ""
    target_id = id_match.group(1)
    db_url = f"https://db.netkeiba.com/jockey/{target_id}/" if is_jockey else f"https://db.netkeiba.com/trainer/{target_id}/"
    try:
        res = session.get(db_url, timeout=5)
        if res.status_code == 200:
            res.encoding = "euc-jp"
            sub_soup = BeautifulSoup(res.text, "html.parser")
            h1_target = sub_soup.select_one("#db_main_box > div > div.db_head_name.fc > div > h1")
            if h1_target:
                name_clean = re.sub(r"\(.*\)|（.*）", "", h1_target.text.strip())
                name_clean = " ".join(name_clean.split()).strip()
                cache_dict[raw_url] = name_clean
                return name_clean
    except: pass
    return ""

print("========================================")
print(f" パート1: 出馬表スクレイピング開始 [設定: 天気={天気} / 馬場={馬場状態}]")
print("========================================")

for race_num in range(1, 13):
    race_no = f"{race_num:02d}"
    race_id = f"{year}02{kai:02d}{nichime:02d}{race_no}"
    print(f"ID: {race_id}")
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    
    session.headers.update({
        "Referer": f"https://race.netkeiba.com/top/race_list.html?kaisai_date={year}{month:02d}{day:02d}"
    })
    print(f"-> {race_num}R を解析中...")

    try:
        response = session.get(url, timeout=5)
        if response.status_code != 200: continue
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "html.parser")

        # レース情報の解析（コース・距離・回り）
        race_data_div = soup.find("div", class_="RaceData01")
        race_info_text = race_data_div.text.strip() if race_data_div else ""
        track_type, direction, distance = "芝", "右", 1200
        if race_info_text:
            track_type = "ダ" if "ダ" in race_info_text else "芝"
            direction = "左" if "左" in race_info_text else ("直" if "直" in race_info_text else "右")
            dist_match = re.search(r"(\d+)m", race_info_text)
            if dist_match: distance = int(dist_match.group(1))

        # レース条件
        race_data_02 = soup.find("div", class_="RaceData02")
        race_condition = "(混)[指]馬齢"
        if race_data_02:
            c_clean = re.sub(r"\d+回[^\d]+\d+日目|(新馬|未勝利|１勝クラス|1勝クラス|２勝クラス|2勝クラス|３勝クラス|3勝クラス|オープン|OP|重賞|GI|G1|GII|G2|GIII|G3|サラ系|アラブ系)|\d+歳(以上|未満)?|\d+頭|本賞金.*$", "", race_data_02.text.strip())
            race_condition = re.sub(r"([\]\)])\s+", r"\1", " ".join(c_clean.split()).strip())
            if not race_condition: race_condition = "定量"

        table = soup.find("table", class_="Shutuba_Table")
        if not table: continue
        rows = table.find_all("tr", class_="HorseList")

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 8: continue

            waku = cols[0].text.strip()
            umaban = cols[1].text.strip()
            horse_a = cols[3].find("a")
            horse_name = horse_a.text.strip() if horse_a else ""
            seirei = cols[4].text.strip()
            seibetsu, age = seirei[0] if seirei else "", seirei[1:] if len(seirei) > 1 else ""
            weight = cols[5].text.strip()
            if weight:
                try: weight = f"{float(weight):.1f}"
                except ValueError: pass

            # 騎手・調教師（フルネーム化）
            jockey_td = cols[6]
            j_full = fetch_real_fullname(jockey_td.find("a")["href"] if jockey_td.find("a") else "", jockey_cache, is_jockey=True)
            jockey = j_full if j_full else re.sub(r"[☆▲△▼◇★▶縲咲┯\s]", "", jockey_td.text.strip())

            trainer_td = cols[7]
            t_full = fetch_real_fullname(trainer_td.find("a")["href"] if trainer_td.find("a") else "", trainer_cache, is_jockey=False)
            trainer = t_full if t_full else trainer_td.text.strip()[2:]
            belongs = "東" if "美浦" in trainer_td.text else ("西" if "栗東" in trainer_td.text else "地")

            # 馬体重・体重増減を出馬表の「450(+2)」から分解抽出
            batai_td = cols[8].text.strip() if len(cols) > 8 else ""
            batai_match = re.search(r"(\d+)\((.*?)\)", batai_td)
            if batai_match:
                batai_val = int(batai_match.group(1))
                zougen_raw = batai_match.group(2)
                try: zougen_val = int(zougen_raw)
                except ValueError: zougen_val = 0
            else:
                batai_val, zougen_val = 0, 0

            odds_text = cols[9].text.strip() if len(cols) > 9 else ""
            odds = odds_text if odds_text and "---" not in odds_text and "計不" not in odds_text else ""

            all_race_data.append({
                "年": int(year), "月": month, "日": day, "曜日": day_of_week, "場所": place_val, "回": int(kai), "日目": int(nichime), "レース目": race_num,
                "天気": 天気, "馬場状態": 馬場状態, "レース条件": race_condition, "芝ダート": track_type, "距離": distance, "回り": direction, "出走数": len(rows),
                "枠番": waku, "馬番": umaban, "馬名": horse_name, "性別": seibetsu, "年齢": int(age) if age.isdigit() else age, "斤量": weight,
                "騎手": jockey, "所属": belongs, "調教師名": trainer, "馬体重": batai_val, "体重増減": zougen_val,
                "過去出走回数": 0, "過去平均着順": 0, "過去連対率": 0, "過去複勝率": 0, "過去平均上がり偏差値": 0, "オッズ": odds,
            })
        time.sleep(1.0)
    except Exception as e:
        print(f"エラー: {e}")

if all_race_data:
    shunba_df = pd.DataFrame(all_race_data)
else:
    print("データが取得できませんでした。処理を終了します。"); exit()

# ==============================================================================
# パート2: 過去スタッツマッピング
# ==============================================================================
print("\n========================================")
print(" パート2: 過去スタッツのマッピングを開始します")
print("========================================")

try:
    past_df = pd.read_csv('processed_10_data.csv', header=None)
    print("-> processed_10_data.csv の読み込みに成功しました。")

    past_df = past_df.sort_values(by=[1, 2, 3, 5, 8], ascending=True) 
    latest_stats = past_df.groupby(20).last().reset_index()

    stats_dict = {}
    for _, row in latest_stats.iterrows():
        stats_dict[str(row[20]).strip()] = row.iloc[-5:].tolist()

    print("-> 直近戦績を結合中...")
    for idx, row in shunba_df.iterrows():
        horse_name = str(row['馬名']).strip()
        if horse_name in stats_dict:
            stats = stats_dict[horse_name]
            shunba_df.at[idx, '過去出走回数'] = stats[0]
            shunba_df.at[idx, '過去平均着順'] = stats[1]
            shunba_df.at[idx, '過去連対率']   = stats[2]
            shunba_df.at[idx, '過去複勝率']   = stats[3]
            shunba_df.at[idx, '過去平均上がり偏差値'] = stats[4]

    columns_order = [
        "年", "月", "日", "曜日", "場所", "回", "日目", "レース目", "天気", "馬場状態", 
        "レース条件", "芝ダート", "距離", "回り", "出走数", "枠番", "馬番", "馬名", 
        "性別", "年齢", "斤量", "騎手", "所属", "調教師名", "馬体重", "体重増減", 
        "過去出走回数", "過去平均着順", "過去連対率", "過去複勝率", "過去平均上がり偏差値", "オッズ"
    ]
    shunba_df = shunba_df.reindex(columns=columns_order)
    
    # 指定された日付入りのファイル名で上書き出力
    shunba_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n【全工程が正常完了しました！】")
    print(f"最終保存ファイル名: {output_filename}")

except FileNotFoundError:
    print(f"\n[警告] processed_10_data.csv が見つからないため、スタッツ結合をスキップしました。")
    shunba_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"スクレイピングデータのみをファイル名 '{output_filename}' に保存しました。")