# import re
import asyncio
# import pandas as pd
# from pathlib import Path
from utils import PopupHandler
from playwright.async_api import async_playwright

if __name__ == "__main__":
    async def main():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            ph = PopupHandler()


            await page.goto("https://www.passports.gov.au/")
            await ph.close_popups(country='Australia', category='passport_application_portal', page=page)
            await ph.scroll_to_bottom(page)
            # print(await page.title())
            await page.screenshot(path="screenshot1.png", full_page=True)
            await browser.close()

asyncio.run(main())