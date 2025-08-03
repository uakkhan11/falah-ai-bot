# live_price_reader.py

import os
import json
import glob

def load_token_map(path="/root/falah-ai-bot/token_map.json"):
    with open(path) as f:
        return json.load(f)  # {symbol: token}

def reverse_token_map(token_map):
    return {v: k for k, v in token_map.items()}  # {token: symbol}

def get_combined_live_prices(tmp_folder="/tmp"):
    prices = {}
    for path in glob.glob(os.path.join(tmp_folder, "live_prices_*.json")):
        try:
            with open(path) as f:
                worker_data = json.load(f)
                prices.update({int(k): v for k, v in worker_data.items()})
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
    return prices  # {token: price}

def get_symbol_price_map():
    token_map = load_token_map()
    reverse_map = reverse_token_map(token_map)
    token_prices = get_combined_live_prices()
    return {
        reverse_map[token]: price
        for token, price in token_prices.items()
        if token in reverse_map
    }
