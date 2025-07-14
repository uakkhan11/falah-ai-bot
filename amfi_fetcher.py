# amfi_fetcher.py

import requests
import pandas as pd
import os
import json
from datetime import datetime
from io import BytesIO
import camelot

OUTPUT_JSON = "/root/falah-ai-bot/large_mid_cap.json"
PDF_URL = "https://www.amfiindia.com/Themes/Theme1/downloads/AverageMarketCapitalization30Jun2025.pdf"
TEMP_PDF = "/root/falah-ai-bot/amfi_large_midcap.pdf"

def fetch_amfi_pdf():
    print("✅ Downloading AMFI Large & Mid Cap PDF...")
    response = requests.get(PDF_URL)
    if response.status_code != 200 or not response.content.startswith(b"%PDF"):
        raise Exception("❌ Invalid PDF content from AMFI site.")
    with open(TEMP_PDF, "wb") as f:
        f.write(response.content)
    print(f"✅ PDF saved at {TEMP_PDF}")
    return TEMP_PDF

def parse_pdf_to_symbols(file_path):
    print("✅ Parsing PDF using Camelot...")
    tables = camelot.read_pdf(file_path, pages='all', flavor='stream')

    symbols = set()
    for table in tables:
        df = table.df
        # skip empty tables
        if df.empty or df.shape[0] < 2:
            continue
        # Clean column headers
        df.columns = [str(c).strip() for c in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)
        # Search all columns for valid NSE symbols
        for col in df.columns:
            for val in df[col].dropna():
                val = str(val).strip()
                if val.isupper() and val.isalpha() and 2 <= len(val) <= 12:
                    symbols.add(val)
    print(f"✅ Found {len(symbols)} unique Large/Mid Cap symbols.")
    return sorted(list(symbols))

def save_to_json(symbols):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(symbols, f, indent=2)
    print(f"✅ Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    try:
        pdf_path = fetch_amfi_pdf()
        symbols = parse_pdf_to_symbols(pdf_path)
        save_to_json(symbols)
        print(f"✅ Large/Mid Cap list successfully updated on {datetime.now().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"❌ Error: {e}")
