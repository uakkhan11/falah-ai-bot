# smart_scanner.py

import os
import json
import pandas as pd
from utils import get_halal_list
from credentials import get_kite, load_secrets
import glob

HIST_DIR = "/root/falah-ai-bot/historical_data/"

def load_all_live_prices():
    live = {}
    for f in glob.glob("/tmp/live_prices_*.json"):
        with open(f) as fd:
            live.update(json.load(fd))
    return live

def run_smart_scan():
    kite = get_kite()
    live_prices = load_all_live_prices()
    with open("/root/falah-ai-bot/tokens.json") as f:
        token_map = json.load(f)
    token_to_symbol = {v: k for k, v in token_map.items()}

    results = []
    for token, ltp in live_prices.items():
        sym = token_to_symbol.get(str(token))
        if not sym:
            continue

        hist_file = os.path.join(HIST_DIR, f"{sym}.csv")
        if not os.path.exists(hist_file):
            continue

        df = pd.read_csv(hist_file)

        # Simple logic: close > SMA(20)
        df["SMA20"] = df["close"].rolling(20).mean()
        last_sma = df["SMA20"].iloc[-1]
        if ltp > last_sma:
            results.append({
                "Symbol": sym,
                "CMP": ltp,
                "SMA20": round(last_sma, 2)
            })

    return pd.DataFrame(results)
