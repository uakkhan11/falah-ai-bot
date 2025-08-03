# instrument_loader.py

import json
from credentials import get_kite

def save_token_map():
    kite = get_kite()
    instruments = kite.instruments()

    token_map = {str(i["instrument_token"]): i["tradingsymbol"] for i in instruments if i["exchange"] == "NSE"}
    reverse_map = {v: int(k) for k, v in token_map.items()}

    with open("instrument_token_map.json", "w") as f:
        json.dump(token_map, f)

    with open("symbol_to_token.json", "w") as f:
        json.dump(reverse_map, f)

    print("✅ Instrument maps saved")

if __name__ == "__main__":
    save_token_map()
