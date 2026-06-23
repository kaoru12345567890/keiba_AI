import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import pandas as pd

async def run():
    async with async_playwright() as p:
        # ブラウザを起動（ヘッドレスモード）
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        all_data = []

        for year in range(2010, 2027):
            url = f"https://db.netkeiba.com/race/list/{year}/"
            print(f"アクセス中: {year}...")
            
            await page.goto(url)
            # JavaScriptの読み込みを待つ
            await page.wait_for_selector('table')
            
            # HTMLを取得して解析
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # 実際の開催データテーブルを探す
            # netkeibaの構造解析に基づき、開催情報を特定
            tables = soup.find_all('table')
            for table in tables:
                # 開催日程が含まれるテーブルを抽出（構造を確認済み）
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        place = cols[0].get_text(strip=True)
                        kai = cols[1].get_text(strip=True)
                        days = cols[2].get_text(strip=True)
                        all_data.append({'Year': year, 'Place': place, 'Kai': kai, 'Days': days})
        
        await browser.close()
        
        # CSV保存
        df = pd.DataFrame(all_data)
        df.to_csv("netkeiba_fixed_data.csv", index=False)
        print("完了: netkeiba_fixed_data.csv に全データを書き込みました。")

if __name__ == "__main__":
    asyncio.run(run())