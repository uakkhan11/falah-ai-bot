# amfi_fetcher.py

import requests
import pandas as pd
import os
import json
from datetime import datetime
from io import BytesIO
import camelot

PDF_URL = "https://www.amfiindia.com/spages/AMFI%20Large%20Mid%20Cap%20Funds.pdf"
PDF_PATH = "/root/falah-ai-bot/amfi_large_midcap.pdf"
OUTPUT_JSON = "/root/falah-ai-bot/large_mid_cap.json"

def fetch_amfi_pdf():
    print("✅ Downloading AMFI Large & Mid Cap PDF...")
    response = requests.get(PDF_URL)
    if response.status_code != 200 or b"%PDF" not in response.content[:10]:
        raise Exception("Failed to download a valid AMFI PDF")
    with open(PDF_PATH, "wb") as f:
        f.write(response.content)
    print(f"✅ PDF saved at {PDF_PATH}")

def parse_pdf_to_symbols():
    print("✅ Parsing PDF using Camelot...")
    tables = camelot.read_pdf(PDF_PATH, pages="all", flavor="stream")

    symbols = set()
    for table in tables:
        df = table.df
        if df.shape[0] < 2:
            continue
        df.columns = df.iloc[0].str.strip()
        df = df.iloc[1:]
        for col in df.columns:
            matches = df[col].dropna().unique()
            for m in matches:
                m = str(m).strip()
                if m.isupper() and m.isalpha() and 2 <= len(m) <= 12:
                    symbols.add(m)

    print(f"✅ Found {len(symbols)} unique Large/Mid Cap NSE symbols.")
    return sorted(symbols)

def save_symbols(symbols):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(symbols, f, indent=2)
    print(f"✅ Saved Large/Mid Cap list to {OUTPUT_JSON}")

if __name__ == "__main__":
    try:
        fetch_amfi_pdf()
        symbols = parse_pdf_to_symbols()
        save_symbols(symbols)
        print(f"✅ Large/Mid Cap list successfully updated on {datetime.now().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"❌ Error: {e}")
