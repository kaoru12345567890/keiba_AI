import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# ==============================================================================
# 設定：ファイルパス
# ==============================================================================
file_path = r'C:\keiba_AI\newgettoday\hakodate_2026020111.csv'

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

raw_input = "2026020111"
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