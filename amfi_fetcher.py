import requests
import pandas as pd
import os
import json
from datetime import datetime
from io import BytesIO
import camelot

# Output JSON file
OUTPUT_JSON = "/root/falah-ai-bot/large_mid_cap.json"

# AMFI Large Midcap PDF URL (updated every 6 months by SEBI)
PDF_URL = "https://www.amfiindia.com/spages/AMFI%20Large%20Mid%20Cap%20Funds.pdf"

def fetch_amfi_pdf():
    print("✅ Downloading AMFI Large & Mid Cap PDF...")
    response = requests.get(PDF_URL)
    if response.status_code != 200:
        raise Exception("Failed to download AMFI PDF")
    return BytesIO(response.content)

def parse_pdf_to_symbols(pdf_data):
    print("✅ Parsing PDF using Camelot...")
    tables = camelot.read_pdf(filepath_or_buffer=pdf_data, pages='all', flavor='stream')

    symbols = set()
    for table in tables:
        df = table.df
        # Clean all columns
        df.columns = [c.strip() for c in df.iloc[0]]
        df = df.iloc[1:]
        for col in df.columns:
            matches = df[col].dropna().unique()
            for m in matches:
                m = str(m).strip()
                # Extract NSE symbol assumption (all caps, no spaces)
                if m.isupper() and m.isalpha() and len(m) <= 12:
                    symbols.add(m)
    print(f"✅ Found {len(symbols)} unique Large/Mid Cap symbols from PDF.")
    return sorted(list(symbols))

def save_to_json(symbols):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(symbols, f, indent=2)
    print(f"✅ Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    try:
        pdf_data = fetch_amfi_pdf()
        symbols = parse_pdf_to_symbols(pdf_data)
        save_to_json(symbols)
        print(f"✅ Large/Mid Cap list successfully updated on {datetime.now().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"❌ Error: {e}")
