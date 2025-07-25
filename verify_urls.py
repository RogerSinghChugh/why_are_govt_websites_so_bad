import asyncio
import httpx
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
# from urllib.parse import urlparse

INPUT_CSV = "resources/gov_sites.csv"
CLEAN_CSV = "resources/gov_sites_clean.csv"
INVALID_CSV = "resources/gov_sites_invalid.csv"
SUMMARY_CSV = "resources/gov_sites_summary_by_country.csv"

MAX_CONCURRENCY = 50
TIMEOUT = 12.0
OK_STATUS_MIN = 200
OK_STATUS_MAX = 399  # consider redirects OK

HEADERS = {
    "User-Agent": "gov-url-checker/1.0 (+https://example.com; contact: you@example.com)"
}

# Some servers reject HEAD; we fallback to GET
async def check_url(client: httpx.AsyncClient, url: str) -> dict:
    try:
        # Try HEAD first
        r = await client.head(url, timeout=TIMEOUT)
        status = r.status_code
        final_url = str(r.url)
        if status < OK_STATUS_MIN or status > OK_STATUS_MAX:
            # Retry with GET if not ok
            r = await client.get(url, timeout=TIMEOUT)
            status = r.status_code
            final_url = str(r.url)
        ok = OK_STATUS_MIN <= status <= OK_STATUS_MAX
        return {"ok": ok, "status": status, "final_url": final_url, "error": None}
    except httpx.HTTPError as e:
        return {"ok": False, "status": None, "final_url": None, "error": repr(e)}
    except Exception as e:
        return {"ok": False, "status": None, "final_url": None, "error": repr(e)}

async def bound_check(semaphore, client, row_idx, url):
    async with semaphore:
        res = await check_url(client, url)
        res["row_idx"] = row_idx
        return res

async def main():
    df = pd.read_csv(INPUT_CSV)
    assert {"country", "category", "url"}.issubset(df.columns), "CSV must have country,category,url columns"

    # Optional: strip spaces
    df["url"] = df["url"].str.strip()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient(
        # headers=HEADERS,
        follow_redirects=True,
        http2=True,  # faster if supported
        verify=True,
    ) as client:
        tasks = [
            bound_check(semaphore, client, i, url)
            for i, url in enumerate(df["url"].tolist())
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Checking URLs")

    res_df = pd.DataFrame(results).set_index("row_idx").sort_index()
    merged = df.join(res_df)

    valid = merged[merged["ok"] == True].copy()
    invalid = merged[merged["ok"] == False].copy()

    # Optional: drop duplicates that resolve to identical final_url
    # (keep the first occurrence)
    valid = valid.drop_duplicates(subset=["final_url"], keep="first")

    valid[["country", "category", "url", "final_url", "status"]].to_csv(CLEAN_CSV, index=False)
    invalid[["country", "category", "url", "status", "error"]].to_csv(INVALID_CSV, index=False)

    summary = (
        valid.groupby("country")
        .size()
        .reset_index(name="valid_count")
        .sort_values("valid_count", ascending=False)
    )
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n=== Done ===")
    print(f"Total rows:        {len(df)}")
    print(f"Valid (kept):      {len(valid)}  -> {CLEAN_CSV}")
    print(f"Invalid (dropped): {len(invalid)} -> {INVALID_CSV}")
    print(f"Summary by country: {SUMMARY_CSV}")

if __name__ == "__main__":
    asyncio.run(main())
