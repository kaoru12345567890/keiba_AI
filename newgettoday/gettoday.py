#@title 【設定】本日のレース環境入力フォーム { run: "auto" }
#@markdown ※右側のメニューを選択してからセルを実行してください。

天気 = "晴" #@param ["晴", "曇", "雨", "小雨", "雪"]
馬場状態 = "良" #@param ["良", "稍重", "重", "不良"]

# ==============================================================================
# 以下、全自動スクレイピングコード（修正版）
# ==============================================================================
import os
import re
import time
import datetime
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# 開催情報を一括入力 (例: 202606130101)
raw_input = input("開催情報を入力してください (例: 202606130101): ")
if len(raw_input) != 12:
    print("エラー: 入力形式が違います。12桁の数字で入力してください。")
    raw_input = input("開催情報を入力してください (例: 202606130101): ")
    if len(raw_input) != 12:
        exit()

# 文字列から各情報を抽出
year = raw_input[0:4]
month = str(int(raw_input[4:6]))
day = str(int(raw_input[6:8]))
kai = str(int(raw_input[8:10]))
nichime = str(int(raw_input[10:12]))

date_obj = datetime.date(int(year), int(month), int(day))
weekday_list = ["月", "火", "水", "木", "金", "土", "日"]
day_of_week = weekday_list[date_obj.weekday()]

# 場所コードに関わらず、場所には一律で「函館」を代入
basho = "函館"

# 入力内容の確認
print("-" * 30)
print(f"日付: {year}年{month}月{day}日 ({day_of_week})")
print(f"開催場所: {basho} (コード: 02)")
print(f"開催回次: 第{kai}回 {nichime}日目")
print("-" * 30)

# ★ 【保存先の設定】指定されたフォルダパスを設定し、なければ自動作成する
output_dir = r"C:\keiba_AI\newgettoday"
os.makedirs(output_dir, exist_ok=True)

# ファイル名生成（フルパス化）
output_filename = os.path.join(output_dir, f"hakodate_{year}02{int(kai):02d}{int(nichime):02d}.csv")

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
    race_id = f"{year}02{int(kai):02d}{int(nichime):02d}{race_no}"
    print(f"ID: {race_id}")
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    
    session.headers.update({
        "Referer": f"https://race.netkeiba.com/top/race_list.html?kaisai_date={year}{int(month):02d}{int(day):02d}"
    })
    print(f"-> {race_num}R を解析中...")

    try:
        response = session.get(url, timeout=5)
        if response.status_code != 200: continue
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "html.parser")

        # h1要素およびdiv要素のRaceNameを確実に取得
        race_name_tag = soup.find(["h1", "div"], class_="RaceName")
        race_name = race_name_tag.text.strip() if race_name_tag else ""
        race_name = re.sub(r'\s+', ' ', race_name)

        # ★ 重賞アイコン（G1=Type1, G2=Type2, G3=Type3）のみを正確にチェック
        if race_name_tag:
            # クラス名が「Icon_GradeType1」「Icon_GradeType2」「Icon_GradeType3」のいずれかと完全一致するか判定（WIN5のType13は除外）
            grade_icon = race_name_tag.find("span", class_=lambda x: x and any(cls in ["Icon_GradeType1", "Icon_GradeType2", "Icon_GradeType3"] for cls in x.split()))
            if grade_icon:
                # アイコンが見つかったら、レース名の先頭に「第0回」を自動で付加する
                race_name = "第0回" + race_name
                print(f"-> 【重賞マーク検出】レース名を変更しました: {race_name}")


        # レース情報の解析（コース・距離・回り）
        race_data_div = soup.find("div", class_="RaceData01")
        race_info_text = race_data_div.text.strip() if race_data_div else ""
        track_type, direction, distance = "芝", "右", "1200"
        if race_info_text:
            track_type = "ダ" if "ダ" in race_info_text else "芝"
            direction = "left" if "左" in race_info_text else ("直線" if "直" in race_info_text else "右")
            dist_match = re.search(r"(\d+)m", race_info_text)
            if dist_match: distance = dist_match.group(1)

        # レース条件
        race_data_02 = soup.find("div", class_="RaceData02")
        race_condition = ""
        if race_data_02:
            c_clean = re.sub(r"\d+回[^\d]+\d+日目|(新馬|未勝利|１勝クラス|1勝クラス|２勝クラス|2勝クラス|３勝クラス|3勝クラス|オープン|OP|重賞|GI|G1|GII|G2|GIII|G3|サラ系|アラブ系)|\d+歳(以上|未満)?|\d+頭|本賞金.*$", "", race_data_02.text.strip())
            race_condition = re.sub(r"([\]\)])\s+", r"\1", " ".join(c_clean.split()).strip())
            if not race_condition: race_condition = "定量"

        # 出馬表テーブルの取得
        table = soup.find("table", class_=lambda x: x and "ShutubaTable" in x or "Shutuba_Table" in x)
        if not table: continue
        rows = table.find_all("tr", class_="HorseList")
        syutsosu = len(rows)

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 8: continue

            wakuban = cols[0].text.strip()
            umaban = cols[1].text.strip()
            horse_a = cols[3].find("a")
            horse_name = horse_a.text.strip() if horse_a else ""
            
            # 性別年齢の取得と分割ロジック
            sei_nen = cols[4].text.strip()
            sei = ""
            nen = ""
            if sei_nen:
                sei = sei_nen[0]    # 1文字目（牡・牝・セ など）
                nen = sei_nen[1:]   # 2文字目以降（年齢の数字）

            kinryo = cols[5].text.strip()

            # 騎手（フルネーム化）
            jockey_td = cols[6]
            j_full = fetch_real_fullname(jockey_td.find("a")["href"] if jockey_td.find("a") else "", jockey_cache, is_jockey=True)
            jockey = j_full if j_full else re.sub(r"[☆▲△▼◇★▶縲咲┯\s]", "", jockey_td.text.strip())

            # 調教師名・所属
            trainer_td = cols[7]
            t_full = fetch_real_fullname(trainer_td.find("a")["href"] if trainer_td.find("a") else "", trainer_cache, is_jockey=False)
            trainer = t_full if t_full else trainer_td.text.strip()[2:]
            shozo = "東" if "美浦" in trainer_td.text else ("西" if "栗東" in trainer_td.text else "地方")

            # 馬体重・体重増減の取得とクリーンアップ
            batai_td = cols[8].text.strip() if len(cols) > 8 else ""
            batai_match = re.search(r"(\d+)\((.*?)\)", batai_td)
            if batai_match:
                bataiju = batai_match.group(1)
                taiju_zougen = batai_match.group(2)
                # プラス記号「+」が含まれている場合は削除（マイナス「-」はそのまま残る）
                taiju_zougen = taiju_zougen.replace("+", "")
            else:
                bataiju = batai_td.replace("計不", "").strip() if batai_td else ""
                taiju_zougen = ""

            # データ詰め込み
            all_race_data.append({
                "レースID": race_id,
                "年": year,
                "月": month,
                "日": day,
                "曜日": day_of_week,
                "場所": basho,
                "回": kai,
                "日目": nichime,
                "レース目": race_num,
                "レース名": race_name,
                "天気": 天気,
                "馬場状態": 馬場状態,
                "レース条件": race_condition,
                "芝ダート": track_type,
                "距離": distance,
                "回り": direction,
                "出走数": syutsosu,
                "枠番": wakuban,
                "馬番": umaban,
                "馬名": horse_name,
                "性別年齢": sei_nen,
                "性別": sei,
                "年齢": nen,
                "斤量": kinryo,
                "騎手": jockey,
                "馬体重": bataiju,
                "体重増減": taiju_zougen,
                "所属": shozo,
                "調教師名": trainer
            })
        time.sleep(1.0)
    except Exception as e:
        print(f"エラー: {e}")

if all_race_data:
    shunba_df = pd.DataFrame(all_race_data)
else:
    print("データが取得できませんでした。処理を終了します。")
    exit()

# ==============================================================================
# パート2: 指定カラムの並び替えとCSV保存
# ==============================================================================
print("\n========================================")
print(" パート2: カラムを指定された29項目に整えます")
print("========================================")

columns_order = [
    "レースID", "年", "月", "日", "曜日", "場所", "回", "日目", "レース目", "レース名",
    "天気", "馬場状態", "レース条件", "芝ダート", "距離", "回り", "出走数", "枠番",
    "馬番", "馬名", "性別年齢", "性別", "年齢", "斤量", "騎手", "馬体重", "体重増減",
    "所属", "調教師名"
]

shunba_df = shunba_df.reindex(columns=columns_order)

# CSV出力
shunba_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n【全工程が正常完了しました！】")
print(f"最終保存ファイル名: {output_filename}")

import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# ==============================================================================
# 設定：ファイルパス
# ==============================================================================
file_path = fr'C:\keiba_AI\newgettoday\hakodate_20260201{nichime}.csv'

if not os.path.exists(file_path):
    print(f"⚠️ 指定のファイルが見つかりません: {file_path}")
    raw_input_file = input("読み込むCSVファイル名を入力してください (例: hakodate_2026020111.csv): ")
    file_path = raw_input_file

df = pd.read_csv(file_path)

# 脚質変換ルール
kyaku_map = {
    "逃": 1.0,
    "先": 2.0,
    "差": 3.0,
    "追": 4.0,
    "全": 5.0
}

raw_input = f"20260201{nichime}"
base_race_id = raw_input.replace("/", "").strip()

# Selenium設定
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=chrome_options)

print("--- 脚質データ抽出・数値変換開始 ---")

for race_num in range(1, 13):
    race_id = f"{base_race_id}{race_num:02d}"
    url = f"https://race.netkeiba.com/race/newspaper.html?m=riot-shutuba-past&race_id={race_id}"
    
    try:
        driver.get(url)
        time.sleep(5) # JavaScript描画待ち
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # 【★修正】_Wrapperを巻き込まないよう、dlタグのHorseListだけを正確に取得
        horse_rows = soup.find_all("dl", class_="HorseList")
        
        found_count = 0
        not_found_count = 0
        
        for row in horse_rows:
            # 本物の馬名が入っているブロックを探す
            name_block = row.find(class_=lambda x: x and "HorseName" in x)
            if not name_block: continue
            
            name_tag = name_block.find("a")
            if not name_tag: continue
            
            # 馬名を取得して余計な空白を排除
            horse_name = "".join(name_tag.text.split())
            
            # 同じ箱の中から「脚質データ（Type）」を探す
            kyaku_val = 0.0
            type_element = row.find(class_=lambda x: x and "Type" in x)
            
            if type_element:
                txt = type_element.text.strip()
                if txt in kyaku_map:
                    kyaku_val = kyaku_map[txt]
                
            # CSVの該当馬へ反映
            mask = (df['レース目'] == race_num) & (df['馬名'] == horse_name)
            if mask.any():
                df.loc[mask, '脚質ラベル'] = kyaku_val
                found_count += 1
                
                if kyaku_val == 0.0:
                    not_found_count += 1
        
        print(f"{race_num}R: {found_count}頭分を処理完了 (うち脚質データなし(0): {not_found_count}頭)")
        
    except Exception as e:
        print(f"{race_num}R でエラー: {e}")

driver.quit()

# 脚質ラベル列を確実に数値型（浮動小数点）にする
df['脚質ラベル'] = df['脚質ラベル'].fillna(0.0).astype(float)

# CSV上書き保存
df.to_csv(file_path, index=False, encoding='utf-8-sig')
print(f"\n--- 完了: {file_path} の脚質ラベルが更新されました ---")