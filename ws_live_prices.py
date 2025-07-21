# ws_live_prices.py

import os
import json
import subprocess
from kiteconnect import KiteConnect
from credentials import load_secrets

def start_all_websockets():
    # ✅ Load secrets
    secrets = load_secrets()
    api_key = secrets["zerodha"]["api_key"]
    access_token = secrets["zerodha"]["access_token"]

    # ✅ Validate access token
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    try:
        profile = kite.profile()
        print(f"✅ Zerodha Token Validated | User: {profile['user_name']}")
    except Exception as e:
        print(f"❌ Invalid access token: {e}")
        raise Exception("Access token invalid or expired. Please refresh before starting websockets.")

    # ✅ Load tokens from token_map.json
    token_file = "/root/falah-ai-bot/token_map.json"
    if not os.path.exists(token_file):
        raise FileNotFoundError(f"Token file not found: {token_file}. Please generate token_map.json first.")

    with open(token_file) as f:
        token_map = json.load(f)

    tokens = list(token_map.values())
    if not tokens:
        raise ValueError("Token list is empty. Please check token_map.json content.")

    # ✅ Split tokens into batches (Kite max limit ~500 per websocket)
    batch_size = 300
    batches = [tokens[i:i + batch_size] for i in range(0, len(tokens), batch_size)]
    print(f"✅ {len(batches)} websocket batch(es) prepared (Batch size = {batch_size}).")

    # ✅ Launch websocket workers
    for i, batch in enumerate(batches, start=1):
        token_str = ",".join(str(t) for t in batch)
        print(f"▶️ Starting websocket batch {i} with {len(batch)} tokens.")

        env = dict(os.environ, ZERODHA_TOKEN=access_token)
        subprocess.Popen([
            "python3", "/root/falah-ai-bot/ws_worker.py",
            api_key, access_token, token_str
        ], env=env)

if __name__ == "__main__":
    start_all_websockets()
