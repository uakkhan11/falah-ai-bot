#!/usr/bin/env python3
import os
import pandas as pd
from datetime import datetime, timedelta

# Reuse BASE_DIR/swing_data if available
try:
    from improved_fetcher import BASE_DIR, DATA_DIRS
except Exception:
    BASE_DIR = "/root/falah-ai-bot"
    DATA_DIRS = {"daily": os.path.join(BASE_DIR, "swing_data")}

OUT_DIR = DATA_DIRS["daily"]
os.makedirs(OUT_DIR, exist_ok=True)

# Replace this block with your actual data source if you have Kite or any API handy.
# For now, this example loads an existing symbol CSV to derive dates and mocks the index close
# if you don't have an API connected. Prefer replacing with real index data fetch.
def build_from_existing_symbol(symbol_csv_path, out_csv):
    df = pd.read_csv(symbol_csv_path)
    if "date" not in df.columns or "close" not in df.columns:
        raise SystemExit("Source CSV must have 'date' and 'close' columns.")
    idx = df[["date","close"]].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    # If this is a placeholder, keep the 'close' from the source symbol just to enable the gate wiring.
    # Replace with real NIFTY index closes via your API for production use.
    idx.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(idx)} rows")

if __name__ == "__main__":
    # Option 1: point to a highly liquid symbol CSV only to mirror dates/structure (temporary).
    # Replace with a real NIFTY index CSV fetch when available.
    sample_csv = os.path.join(OUT_DIR, "RELIANCE.csv")
    out_csv = os.path.join(OUT_DIR, "nifty_50.csv")
    if not os.path.exists(sample_csv):
        raise SystemExit(f"Sample symbol CSV not found at {sample_csv}. Run improved_fetcher.py first or point to another CSV.")
    build_from_existing_symbol(sample_csv, out_csv)
