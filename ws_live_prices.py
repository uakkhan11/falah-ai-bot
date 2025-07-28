# ws_live_prices.py
import os
import json
from kiteconnect import KiteConnect
from credentials import load_secrets
import subprocess

def start_all_websockets():
    secrets = load_secrets()
    api_key = secrets["zerodha"]["api_key"]
    access_token = secrets["zerodha"]["access_token"]

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    try:
        profile = kite.profile()
        print(f"✅ Zerodha Token Validated | User: {profile['user_name']}")
    except Exception as e:
        print(f"❌ Invalid access token: {e}")
        raise

    token_file = "/root/falah-ai-bot/token_map.json"
    if not os.path.exists(token_file):
        raise FileNotFoundError(f"Missing token map: {token_file}")

    with open(token_file) as f:
        token_map = json.load(f)

    tokens = list(token_map.values())
    if not tokens:
        raise ValueError("No tokens found in token_map.json")

    # ✅ Convert token list to comma-separated string
    token_str = ",".join(str(t) for t in tokens)

    print(f"▶️ Launching single WebSocket with {len(tokens)} tokens...")

    env = dict(os.environ, ZERODHA_TOKEN=access_token)

    subprocess.Popen([
        "python3", "/root/falah-ai-bot/ws_worker.py",
        api_key, access_token, token_str
    ], env=env)

if __name__ == "__main__":
    start_all_websockets()
