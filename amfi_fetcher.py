# amfi_fetcher.py

import pandas as pd
import requests
import json
import os
from datetime import datetime

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
OUTPUT_FILE = "/root/falah-ai-bot/large_mid_cap.json"
REFRESH_MONTH = datetime.now().strftime("%Y-%m")

def fetch_amfi_data():
    print("✅ Fetching AMFI data...")
    response = requests.get(AMFI_URL)
    response.raise_for_status()

    data = response.text
    lines = data.splitlines()

    large_mid_cap_stocks = set()
    current_category = None

    for line in lines:
        if "##" in line:
            category = line.replace("##", "").strip()
            if "Large Cap" in category or "Mid Cap" in category:
                current_category = category
            else:
                current_category = None
        elif current_category and line and not line.startswith("Scheme Code"):
            parts = line.split(";")
            if len(parts) >= 4:
                stock_name = parts[3].strip()
                if stock_name:
                    large_mid_cap_stocks.add(stock_name.upper())

    print(f"✅ Found {len(large_mid_cap_stocks)} unique Large and Mid Cap stocks.")

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "updated_on": REFRESH_MONTH,
            "stocks": sorted(list(large_mid_cap_stocks))
        }, f, indent=2)

    print(f"✅ Saved Large/Mid Cap stocks to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_amfi_data()
