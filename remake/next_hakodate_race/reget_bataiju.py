import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

def update_bataiju_all_races():
    # 1. 開催情報の入力
    raw_input_str = input("開催情報を入力してください (例: 202606130101): ")
    try:
        year = int(raw_input_str[0:4])
        kai = int(raw_input_str[8:10])
        nichime = int(raw_input_str[10:12])
    except (ValueError, IndexError):
        print("入力形式が正しくありません。")
        return
    
    # 保存先の定義
    output_filename = f"hakodate_{year}02{kai:02d}{nichime:02d}.csv"
    
    # ファイル存在確認
    if not os.path.exists(output_filename):
        print(f"エラー: {output_filename} が存在しません。")
        return

    # DataFrame読み込み
    shunba_df = pd.read_csv(output_filename)
    
    # セッションの初期化
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    print(f"-> 処理を開始します。")
    print(f"-> 保存先ファイル: {os.path.abspath(output_filename)}")

    # 2. 全レースをループ
    for race_target in range(1, 13):
        race_id = f"{year}02{kai:02d}{nichime:02d}{race_target:02d}"
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        
        try:
            response = session.get(url, timeout=10)
            response.encoding = 'euc-jp'
            soup = BeautifulSoup(response.text, "html.parser")
            
            # 馬リストの取得
            table = soup.find("table", class_="Shutuba_Table")
            if not table:
                print(f"-> {race_target}R: データが見つかりません。")
                continue
            
            rows = table.find_all("tr", class_="HorseList")
            print(f"-> {race_target}R: {len(rows)}頭の馬を確認しました。")
            
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 9: continue
                
                umaban_text = cols[1].text.strip()
                batai_text = cols[8].text.strip()
                
                if not umaban_text.isdigit(): continue
                umaban = int(umaban_text)
                
                # 馬体重と増減の抽出
                match = re.search(r"(\d+)\(([-＋＋－\d]+)\)", batai_text)
                if match:
                    weight = int(match.group(1))
                    diff_raw = match.group(2).replace("＋", "+").replace("－", "-")
                    try:
                        diff = int(diff_raw)
                    except:
                        diff = 0
                    
                    # データ反映
                    mask = (shunba_df['レース目'].astype(int) == race_target) & (shunba_df['馬番'].astype(int) == umaban)
                    if mask.any():
                        shunba_df.loc[mask, '馬体重'] = weight
                        shunba_df.loc[mask, '体重増減'] = diff
            
            time.sleep(1.5)
            
        except Exception as e:
            print(f"-> {race_target}R: 解析中にエラーが発生しました: {e}")
            continue

    # 3. ファイルの上書き保存
    shunba_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    # 完了報告と保存先の詳細
    print("="*40)
    print("【完了】馬体重データの更新が終了しました。")
    print(f"保存先: {os.path.abspath(output_filename)}")
    print(f"更新日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*40)

if __name__ == "__main__":
    update_bataiju_all_races()