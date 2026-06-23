import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# --- 設定 ---
raw_input = input("IDを入れてね(例: 20260201/01←多分ここだけ変える): ")
file_path = f"hakodate_{raw_input}.csv"
df = pd.read_csv(file_path)

# 脚質変換ルール: 該当なしはすべて0.0
kyaku_map = {
    "逃": 1.0,
    "先": 2.0,
    "差": 3.0,
    "追": 4.0
}

# Selenium設定
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=chrome_options)

print("--- 脚質データ抽出・数値変換開始 ---")

for race_num in range(1, 13):
    race_id = f"{raw_input}{race_num:02d}"
    url = f"https://race.netkeiba.com/race/newspaper.html?m=riot-shutuba-past&race_id={race_id}"
    
    try:
        driver.get(url)
        time.sleep(5) # JavaScript描画待ち
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        horse_elements = soup.find_all("dt", class_="Horse_Info")
        
        found_count = 0
        for element in horse_elements:
            name_tag = element.find("a")
            if not name_tag: continue
            
            horse_name = name_tag.text.strip()
            
            # 脚質要素を取得
            kyaku_span = element.select_one("dl dt.Horse06.fc div span")
            
            # ラベルを取得して変換。なければ0.0
            if kyaku_span:
                raw_label = kyaku_span.text.strip()
                kyaku_val = kyaku_map.get(raw_label, 0.0)
            else:
                kyaku_val = 0.0
                
            # CSVの該当馬へ反映
            mask = (df['レース目'] == race_num) & (df['馬名'] == horse_name)
            if mask.any():
                df.loc[mask, '脚質ラベル'] = kyaku_val
                found_count += 1
        
        print(f"{race_num}R: {found_count}頭分を処理完了")
        
    except Exception as e:
        print(f"{race_num}R でエラー: {e}")

driver.quit()

# 脚質ラベル列を確実に数値型（浮動小数点）にする
df['脚質ラベル'] = df['脚質ラベル'].fillna(0.0).astype(float)

# CSV上書き保存
df.to_csv(file_path, index=False, encoding='utf-8-sig')
print("--- 完了: 脚質ラベルが 0.0〜4.0 で更新されました ---")