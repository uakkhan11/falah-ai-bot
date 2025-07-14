# amfi_fetcher.py

import pandas as pd
import json

# ✅ Static CSV File containing 'Symbol', 'Company Name', 'Category' (Large/Mid)
CSV_FILE = "/root/falah-ai-bot/amfi_large_mid_cap.csv"
OUTPUT_JSON = "/root/falah-ai-bot/large_mid_cap.json"

def load_large_mid_caps():
    df = pd.read_csv(CSV_FILE)
    df = df[df['Category'].isin(['Large Cap', 'Mid Cap'])]
    symbols = df['Symbol'].dropna().unique().tolist()
    print(f"✅ Found {len(symbols)} unique Large and Mid Cap stocks.")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(symbols, f, indent=2)
    print(f"✅ Saved Large/Mid Cap stocks to {OUTPUT_JSON}")

def get_large_mid_caps():
    with open(OUTPUT_JSON) as f:
        return json.load(f)

if __name__ == "__main__":
    print("✅ Loading Large/Mid Cap symbols from static CSV...")
    load_large_mid_caps()
