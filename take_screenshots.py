"""Take viewport screenshots of every portal listed in resources/websites.xlsx.

Usage:
    python take_screenshots.py                      # all rows -> resources/pictures
    python take_screenshots.py --countries india,brazil --out resources/pictures
    python take_screenshots.py --headless --settle 8

Each row produces <country_slug>_<n>.png where n is 1 = tax identifier, 2 = tax e-filing, 3 = passport.
Existing files with the same name are overwritten.
"""

import argparse
import asyncio
import re
from pathlib import Path

import pandas as pd
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from utils import PopupHandler

EXCEL_PATH = "resources/websites.xlsx"
OUTPUT_DIR = Path("resources/pictures")
MAX_CONCURRENCY = 2
VIEWPORT = {"width": 1920, "height": 1080}
GOTO_TIMEOUT_MS = 60_000
NETWORK_IDLE_TIMEOUT_MS = 15_000

CAT_INDEX = {
    "national_tax_identifier_portal": 1,
    "income_tax_efiling_portal": 2,
    "passport_application_portal": 3,
}


def slugify_country(country: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(country).strip().lower()).strip("_")


def safe_filename(country: str, category: str) -> str:
    return f"{slugify_country(country)}_{CAT_INDEX.get(category, 'X')}.png"


def read_rows(excel_path: str, countries=None, categories=None):
    df = pd.read_excel(excel_path)
    if countries:
        wanted = {slugify_country(c) for c in countries}
        df = df[df["country"].map(slugify_country).isin(wanted)]
    if categories:
        df = df[df["category"].isin(categories)]
    return df.to_dict("records")


async def screenshot_one(entry, browser, ph: PopupHandler, semaphore, out_dir: Path, settle_ms: int):
    url, country, category = entry["website_url"], entry["country"], entry["category"]
    target = out_dir / safe_filename(country, category)
    label = f"{country} | {category}"

    async with semaphore:
        context = await browser.new_context(viewport=VIEWPORT, ignore_https_errors=False)
        page = await context.new_page()
        try:
            print(f"[..] {label} -> {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT_MS)
            try:
                await page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
            except PlaywrightTimeoutError:
                print(f"    network never went idle for {label}; continuing")

            # pass 1: banners that are already up
            await ph.close_popups(country=country, category=category, page=page)
            # let late scripts, lazy images and delayed banners settle
            await page.wait_for_timeout(settle_ms)
            # pass 2: banners or modals that appeared during the settle wait
            await ph.close_popups(country=country, category=category, page=page)
            await page.wait_for_timeout(500)

            await page.screenshot(path=str(target), full_page=False)
            print(f"[OK] {label} -> {target}")
            return {"label": label, "ok": True, "file": str(target)}
        except Exception as e:
            print(f"[ERR] {label} | {url} -> {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            return {"label": label, "ok": False, "error": str(e)}
        finally:
            await context.close()


async def run(args):
    rows = read_rows(args.excel, args.countries, args.categories)
    if not rows:
        print("No rows to process.")
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} row(s) -> {out_dir}  (headless={args.headless}, concurrency={args.concurrency}, settle={args.settle}s)")

    ph = PopupHandler()
    semaphore = asyncio.Semaphore(args.concurrency)

    async with async_playwright() as p:
        # AutomationControlled off makes bot checks (Cloudflare Turnstile) far less likely to trigger;
        # --channel chrome uses the installed Google Chrome instead of Playwright's Chromium build.
        browser = await p.chromium.launch(
            headless=args.headless,
            channel=args.channel,
            args=["--disable-blink-features=AutomationControlled"],
        )
        tasks = [screenshot_one(r, browser, ph, semaphore, out_dir, int(args.settle * 1000)) for r in rows]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await browser.close()

    ok = [r for r in results if isinstance(r, dict) and r.get("ok")]
    bad = [r for r in results if not (isinstance(r, dict) and r.get("ok"))]
    print(f"\nDone: {len(ok)} ok, {len(bad)} failed")
    for r in bad:
        print("  FAILED:", r.get("label") if isinstance(r, dict) else repr(r))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excel", default=EXCEL_PATH)
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    ap.add_argument("--countries", type=lambda s: [c.strip() for c in s.split(",") if c.strip()], default=None,
                    help="comma-separated country names to include (default: all)")
    ap.add_argument("--categories", type=lambda s: [c.strip() for c in s.split(",") if c.strip()], default=None,
                    help="comma-separated category keys to include (default: all)")
    ap.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    ap.add_argument("--settle", type=float, default=10.0, help="seconds to wait after load before the shot")
    ap.add_argument("--headless", action="store_true", help="run Chromium headless (default: headed)")
    ap.add_argument("--channel", default=None, help="browser channel, e.g. 'chrome' or 'msedge' (default: bundled Chromium)")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
