import os
import time
import requests
import pandas as pd
from datetime import date, timedelta
from tqdm import tqdm

DATA_FOLDER = "./data"
os.makedirs(DATA_FOLDER, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.nseindia.com"
})


def get_trading_days(start, end):
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri only
            days.append(current)
        current += timedelta(days=1)
    return days


def download_bhavcopy(dt):
    filename = f"BhavCopy_{dt.strftime('%d%b%Y').upper()}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)

    if os.path.exists(filepath):
        return "exists"

    # NSE archive URL format
    url = (
        f"https://archives.nseindia.com/content/historical/EQUITIES/"
        f"{dt.year}/{dt.strftime('%b').upper()}/"
        f"cm{dt.strftime('%d%b%Y').upper()}bhav.csv.zip"
    )

    try:
        resp = SESSION.get(url, timeout=15)
        if resp.status_code == 200:
            # Save zip and extract
            zip_path = filepath + ".zip"
            with open(zip_path, "wb") as f:
                f.write(resp.content)
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DATA_FOLDER)
            os.remove(zip_path)
            return "downloaded"
        else:
            return f"skip_{resp.status_code}"
    except Exception as e:
        return f"error_{str(e)[:30]}"


# Generate all weekdays Apr 2021 - Mar 2024
start_date = date(2021, 4, 1)
end_date = date(2024, 3, 31)
trading_days = get_trading_days(start_date, end_date)

print(f"Attempting {len(trading_days)} weekdays (holidays will be skipped automatically)\n")

downloaded, skipped, errors = 0, 0, 0

for dt in tqdm(trading_days, desc="Downloading"):
    result = download_bhavcopy(dt)
    if result == "downloaded":
        downloaded += 1
    elif result == "exists":
        skipped += 1
    elif result.startswith("skip"):
        skipped += 1  # holiday or non-trading day
    else:
        errors += 1
    time.sleep(0.3)  # be polite to NSE servers

print(f"\nDone!")
print(f"  Downloaded : {downloaded}")
print(f"  Skipped    : {skipped} (already existed or holidays)")
print(f"  Errors     : {errors}")

files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
print(f"  Total CSVs : {len(files)}")