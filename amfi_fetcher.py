import pandas as pd
import json

EXCEL_FILE = "/root/falah-ai-bot/amfi_large_mid_cap.xlsx"
OUTPUT_JSON = "/root/falah-ai-bot/large_mid_cap.json"

def convert_excel_to_json():
    df = pd.read_excel(EXCEL_FILE)
    # Assuming your Excel has a column named "Symbol"
    symbols = df["Symbol"].dropna().unique().tolist()
    symbols = [str(s).strip().upper() for s in symbols if s]
    print(f"✅ Loaded {len(symbols)} symbols from Excel.")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(symbols, f, indent=2)
    print(f"✅ Saved {len(symbols)} symbols to {OUTPUT_JSON}")

if __name__ == "__main__":
    convert_excel_to_json()
