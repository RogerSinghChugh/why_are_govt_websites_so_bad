import re
import asyncio
import pandas as pd
from pathlib import Path
from utils import PopupHandler
from playwright.async_api import async_playwright


# -------- Config --------
EXCEL_PATH = "resources/websites.xlsx"               
OUTPUT_DIR = Path("resources/pictures")
MAX_CONCURRENCY = 2                        
# NAV_TIMEOUT_MS = 45_000                    # 45s page goto timeout
# IDLE_WAIT_MS = 10_000                      # extra wait for network idle (if it happens)

CAT_INDEX = {
    "national_tax_identifier_portal": 1,
    "income_tax_efiling_portal": 2,
    "passport_application_portal": 3,
}

def slugify_country(country: str) -> str:
    """lowercase + safe filename tokens"""
    return re.sub(r"[^a-z0-9]+", "_", str(country).strip().lower()).strip("_")

def safe_filename(country: str, category: str) -> str:
    n = CAT_INDEX.get(category)
    if not n:
        n = "X"  # unknown category (won't collide with 1/2/3)
    return f"{slugify_country(country)}_{n}.png"

def read_rows(excel_path: str):
    df = pd.read_excel(excel_path)
    # # optional: filter to only our 3 categories
    # df = df[df["category"].isin(CAT_INDEX.keys())].copy()
    rows = df.to_dict("records")
    return rows

async def screenshot_one(entry, browser, ph: PopupHandler, semaphore: asyncio.Semaphore):
    """Open a fresh context → navigate → handle popups → screenshot → close."""
    url = entry["website_url"]
    country = entry["country"]
    category = entry["category"]
    out_name = safe_filename(country, category)
    out_path = OUTPUT_DIR / out_name

    async with semaphore:
        context = await browser.new_context(
            viewport={"width": 1366, "height": 2000},
            # Keeping headless True by default; change in launch() if needed
            ignore_https_errors=False
        )
        page = await context.new_page()
        try:
            # navigate
            await page.goto(url)
            # await page.wait_for_load_state("networkidle")

            # popups/cookies
            await ph.close_popups(country=country, category=category, page=page)

            # expand page before full-page screenshot
            await ph.scroll_to_bottom(page)

            # ensure output dir exists
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # if name collides (duplicate country+category rows), de-dup with a suffix
            target = out_path
            if target.exists():
                stem = target.stem
                ext = target.suffix
                k = 2
                while target.exists():
                    target = target.with_name(f"{stem}-{k}{ext}")
                    k += 1

            await page.screenshot(path=str(target), full_page=True)
            print(f"[OK] {country} | {category} → {target}")
        except Exception as e:
            print(f"[ERR] {country} | {category} | {url} → {e}")
        finally:
            await context.close()

async def main(excel_path: str = EXCEL_PATH, max_concurrency: int = MAX_CONCURRENCY):
    rows = read_rows(excel_path)
    if not rows:
        print("No rows to process.")
        return

    ph = PopupHandler()
    semaphore = asyncio.Semaphore(max_concurrency)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        tasks = [
            asyncio.create_task(screenshot_one(entry, browser, ph, semaphore))
            for entry in rows
        ]
        # gather with return_exceptions so one failure doesn't cancel the rest
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await browser.close()
        # surface unexpected exceptions (optional)
        unexpected = [r for r in results if isinstance(r, Exception)]
        if unexpected:
            print(f"Completed with {len(unexpected)} task errors.")

if __name__ == "__main__":
    asyncio.run(main())

# if __name__ == "__main__":
#     async def main():
#         async with async_playwright() as p:
#             browser = await p.chromium.launch()
#             page = await browser.new_page()
#             ph = PopupHandler()


#             await page.goto("https://mytax.hasil.gov.my/")
#             await ph.close_malysia_tax_identifier_popups(page=page)
#             # print(await page.title())
#             await page.screenshot(path="screenshot1.png", full_page=True)
#             await browser.close()
#
# asyncio.run(main())
