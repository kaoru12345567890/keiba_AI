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
        time.sleep(random.uniform(0.5, 1.2)) # マナーを守った待機
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
        race_data += race_date.contents[0].string.strip().split('/')
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
        return race_data
    except:
        return []

def WebData2Pandas(race_table, race_data):
    rows = []
    for tr in race_table.find_all('tr')[1:]:
        horse_data = [td.get_text(strip=True).replace('\n', '') for td in tr.find_all('td')]
        # 値がない場合の補完
        horse_data = [val if val != '' else (0 if idx == (len(horse_data)-1) else 'NoData') for idx, val in enumerate(horse_data)]
        rows.append(race_data + horse_data)
    return pd.DataFrame(rows)

def fetch_and_save(year, place, number, day, race):
    table, page, race_ID = GetWebPageTable(year, place, number, day, race)
    if table is None:
        return False
    
    place_name = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"][place-1]
    file_path = f'regetData/{place_name}/{year}.csv'
    
    # 重複チェック用ID生成
    with write_lock:
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, dtype={0: str}, usecols=[0])
            if race_ID in existing_df.iloc[:, 0].values:
                return True # 保存済みならスキップ

    race_data = CommonData2List(page, table, race_ID, place, number, day, race)
    if not race_data:
        return False
    
    df = WebData2Pandas(table, race_data)
    
    with write_lock:
        os.makedirs(f'regetData/{place_name}', exist_ok=True)
        df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path), encoding='utf_8_sig')
    print(f"Saved: {race_ID}")
    return True

if __name__ == "__main__":
    for year in range(2026, 2030):
        print(f"--- {year}年のデータ取得を開始します ---")
        for place in range(1, 11):
            for number in range(1, 13):
                if not fetch_and_save(year, place, number, 1, 1): break
                for day in range(1, 13):
                    if not fetch_and_save(year, place, number, day, 1): break
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        futures = [executor.submit(fetch_and_save, year, place, number, day, r) for r in range(1, 13)]
                        for _ in as_completed(futures): pass
        print(f"--- {year}年のデータ取得が完了しました ---")