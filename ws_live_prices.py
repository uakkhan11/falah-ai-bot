# ws_live_prices.py

import json
import subprocess
from credentials import load_secrets
from utils import get_halal_list

def start_all_websockets():
    # Load your Halal tokens only
    with open("/root/falah-ai-bot/tokens.json") as f:
        token_map = json.load(f)

    tokens = list(token_map.values())

    # Split into batches of 300
    batches = [tokens[i:i+300] for i in range(0, len(tokens), 300)]
    print(f"✅ Splitting into {len(batches)} batches of Halal tokens.")

    secrets = load_secrets()
    for i, batch in enumerate(batches, start=1):
        token_str = ",".join(str(t) for t in batch)
        subprocess.Popen([
            "python3", "/root/falah-ai-bot/ws_worker.py",
            secrets["zerodha"]["api_key"],
            secrets["zerodha"]["access_token"],
            token_str
        ])

if __name__ == "__main__":
    start_all_websockets()
