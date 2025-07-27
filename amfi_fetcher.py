import pandas as pd
import json
import os

EXCEL_FILE = "/root/falah-ai-bot/amfi_large_mid_cap.xlsx"
OUTPUT_JSON = "/root/falah-ai-bot/large_mid_cap.json"

def convert_excel_to_json():
    if not os.path.exists(EXCEL_FILE):
        print(f"❌ Excel file not found: {EXCEL_FILE}")
        return

    df = pd.read_excel(EXCEL_FILE)

    if "Symbol" not in df.columns:
        print("❌ 'Symbol' column not found in Excel.")
        return

    symbols = df["Symbol"].dropna().unique().tolist()
    symbols = [str(s).strip().upper() for s in symbols if s]

    print(f"✅ Loaded {len(symbols)} symbols from Excel.")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"symbols": symbols}, f, indent=2)

    print(f"✅ Saved {len(symbols)} symbols to {OUTPUT_JSON}")


def load_large_midcap_symbols():
    if not os.path.exists(OUTPUT_JSON):
        print(f"❌ JSON file not found: {OUTPUT_JSON}")
        return []

    try:
        with open(OUTPUT_JSON, "r") as f:
            data = json.load(f)
        return data.get("symbols", [])
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return []


if __name__ == "__main__":
    convert_excel_to_json()
