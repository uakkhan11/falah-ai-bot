import requests
import pandas as pd
import os
import json
from datetime import datetime
import camelot

PDF_URL = "https://www.amfiindia.com/spages/AMFI%20Large%20Mid%20Cap%20Funds.pdf"
PDF_FILE = "/root/falah-ai-bot/amfi_large_midcap.pdf"
OUTPUT_JSON = "/root/falah-ai-bot/large_mid_cap.json"

def fetch_amfi_pdf():
    print("✅ Downloading AMFI Large & Mid Cap PDF...")
    response = requests.get(PDF_URL)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF, status code {response.status_code}")
    with open(PDF_FILE, "wb") as f:
        f.write(response.content)
    print(f"✅ PDF saved at {PDF_FILE}")

def parse_pdf_to_symbols(file_path):
    print("✅ Parsing PDF using Camelot...")
    tables = camelot.read_pdf(file_path, pages='all', flavor='stream')

    symbols = set()
    for table in tables:
        df = table.df
        for row in df.values:
            for cell in row:
                cell = str(cell).strip()
                if cell.isupper() and cell.isalpha() and len(cell) <= 12:
                    symbols.add(cell)

    print(f"✅ Found {len(symbols)} unique Large/Mid Cap symbols.")
    return sorted(symbols)

def save_to_json(symbols):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(symbols, f, indent=2)
    print(f"✅ Saved symbols to {OUTPUT_JSON}")

if __name__ == "__main__":
    try:
        fetch_amfi_pdf()
        symbols = parse_pdf_to_symbols(PDF_FILE)
        save_to_json(symbols)
        print(f"✅ Large/Mid Cap list successfully updated on {datetime.now().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"❌ Error: {e}")
