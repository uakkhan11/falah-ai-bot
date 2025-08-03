# ws_live_prices.py

import os
import json
import math
import subprocess
import time
from kiteconnect import KiteConnect
from credentials import load_secrets

MAX_TOKENS_PER_SOCKET = 4000
WORKER_PATH = "/root/falah-ai-bot/ws_worker.py"
TOKEN_DIR = "/root/falah-ai-bot/ws_tokens"

def start_all_websockets():
    # Load credentials
    secrets = load_secrets()
    api_key = secrets["zerodha"]["api_key"]
    access_token = secrets["zerodha"]["access_token"]

    # Validate login
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    try:
        profile = kite.profile()
        print(f"✅ Zerodha Token Validated | User: {profile['user_name']}")
    except Exception as e:
        print(f"❌ Invalid access token: {e}")
        raise

    # Load token map
    token_file = "/root/falah-ai-bot/token_map.json"
    if not os.path.exists(token_file):
        raise FileNotFoundError(f"Missing token map: {token_file}")

    with open(token_file) as f:
        token_map = json.load(f)

    tokens = list(token_map.values())
    if not tokens:
        raise ValueError("No tokens found in token_map.json")

    total_tokens = len(tokens)
    print(f"▶️ Launching WebSockets for {total_tokens} tokens...")

    # Create token directory if not exists
    os.makedirs(TOKEN_DIR, exist_ok=True)

    # Launch in slices
    num_sockets = math.ceil(total_tokens / MAX_TOKENS_PER_SOCKET)
    for i in range(num_sockets):
        start = i * MAX_TOKENS_PER_SOCKET
        end = start + MAX_TOKENS_PER_SOCKET
        token_slice = tokens[start:end]
        index = str(i + 1)

        # Save token slice to file
        token_path = os.path.join(TOKEN_DIR, f"ws_tokens_{index}.json")
        with open(token_path, "w") as f:
            json.dump({"tokens": token_slice}, f)

        print(f"🔌 Starting Worker {index}/{num_sockets} for tokens {start}-{end - 1} ({len(token_slice)} tokens)")

        # Launch worker passing only index
        subprocess.Popen([
            "python3", WORKER_PATH, index
        ], env=os.environ.copy())

        time.sleep(1)  # throttle spawn

if __name__ == "__main__":
    start_all_websockets()
