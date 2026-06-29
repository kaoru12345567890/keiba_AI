import os
import time
import random
import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ファイル書き込み時の競合を防ぐロック
write_lock = threading.Lock()
# サーバー負荷対策：3スレッド同時実行
semaphore = threading.BoundedSemaphore(3)

def GetWebPageTable(year, place, number, day, race):
    base_url = 'https://db.sp.netkeiba.com/race/'
    race_ID = f"{year}{str(place).zfill(2)}{str(number).zfill(2)}{str(day).zfill(2)}{str(race).zfill(2)}"
    race_url = base_url + race_ID
    
    try:
        time.sleep(random.uniform(0.5, 1.2))
        with semaphore:
            res = requests.get(race_url, timeout=10)
            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.text, 'lxml')
        
        race_table = soup.find(class_="table_slide_body ResultsByRaceDetail")
        if not race_table or not soup.find(class_="Race_Date"):
            return None, None, race_ID
        return race_table, soup, race_ID
    except:
        return None, None, race_ID

def CommonData2List(race_page, race_table, race_ID, place, number, day, race):
    place_id2name = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
    try:
        race_data = [race_ID]
        race_date = race_page.find_all(class_="Race_Date")[0]
        date_str = race_date.contents[0].string.strip()
        
        # 日付情報から月を抽出
        race_month = int(date_str.split('/')[1])
        
        race_data += date_str.split('/')
        race_data.append(race_date.contents[1].string.strip().replace('(', '').replace(')', ''))
        race_data.append(place_id2name[place-1])
        race_data.append(number)
        race_data.append(day)
        race_data.append(race)
        race_data.append(race_page.find_all(class_="RaceName_main")[0].string)
        race_data.append(race_page.find_all(class_="RaceData")[0].contents[5].contents[0])
        race_data.append(race_page.find_all(class_="RaceData")[0].contents[7].string)
        race_data.append(race_page.find_all(class_="RaceHeader_Value_Others")[0].contents[3].string)
        race_data.append(race_page.find_all(class_="RaceData")[0].contents[3].string[0])
        race_data.append(race_page.find_all(class_="RaceData")[0].contents[3].string[1:6].replace('m', ''))
        race_data.append(race_page.find_all(class_="RaceData")[0].contents[3].string[6:].replace('(', '').replace(')', ''))
        race_data.append(len(race_table.find_all('tr')) - 1)
        
        return race_data, race_month
    except:
        return [], 0

def WebData2Pandas(race_table, race_data):
    rows = []
    for tr in race_table.find_all('tr')[1:]:
        horse_data = [td.get_text(strip=True).replace('\n', '') for td in tr.find_all('td')]
        horse_data = [val if val != '' else (0 if idx == (len(horse_data)-1) else 'NoData') for idx, val in enumerate(horse_data)]
        rows.append(race_data + horse_data)
    return pd.DataFrame(rows)

def fetch_and_save(year, place, number, day, race, start_year, start_month):
    table, page, race_ID = GetWebPageTable(year, place, number, day, race)
    if table is None: return False
    
    race_data, race_month = CommonData2List(page, table, race_ID, place, number, day, race)
    
    # 開始年月以前のデータならスキップ
    if not race_data or (year == start_year and race_month < start_month):
        return False
    
    place_name = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"][place-1]
    file_path = f'regetData/{place_name}/{year}.csv'
    
    with write_lock:
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, dtype={0: str}, usecols=[0])
            if race_ID in existing_df.iloc[:, 0].values: return True

    df = WebData2Pandas(table, race_data)
    with write_lock:
        os.makedirs(f'regetData/{place_name}', exist_ok=True)
        df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path), encoding='utf_8_sig')
    print(f"Saved: {race_ID} ({year}年{race_month}月)")
    return True

if __name__ == "__main__":
    start_month = int(input("開始月を入力してください (例: 1): "))
    start_year = 2026
    end_year = 2030

    for year in range(start_year, end_year + 1):
        print(f"--- {year}年のデータ取得を開始します ---")
        
        for place in range(1, 11):
            fail_count = 0
            for number in range(1, 13):
                for day in range(1, 13):
                    results = []
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        futures = [executor.submit(fetch_and_save, year, place, number, day, r, start_year, start_month) for r in range(1, 13)]
                        for f in as_completed(futures): results.append(f.result())
                    
                    if not any(results): fail_count += 1
                    else: fail_count = 0
                    
                    if fail_count >= 3: break
                if fail_count >= 3: break
                    
        print(f"--- {year}年のデータ取得が完了しました ---")

    print("--- 2026年から2030年までのデータ取得が完了しました ---")