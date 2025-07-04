# credentials.py
import json
import os
import gspread
from datetime import datetime
from kiteconnect import KiteConnect

def load_secrets():
    """
    Load your credentials from the secrets JSON or TOML.
    """
    with open("/root/falah-ai-bot/secrets.json", "r") as f:
        secrets = json.load(f)
    return secrets

def get_kite():
    secrets = load_secrets()
    kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])
    if not os.path.exists("/root/falah-ai-bot/access_token.json"):
        raise Exception("❌ Access token file not found. Please generate it from dashboard UI.")
    with open("/root/falah-ai-bot/access_token.json", "r") as f:
        token = json.load(f)["access_token"]
    kite.set_access_token(token)
    return kite

def validate_kite(kite):
    """
    Check if access token is still valid by calling profile.
    """
    try:
        profile = kite.profile()
        print("✅ Valid credentials loaded.")
        return True
    except Exception as e:
        print(f"❌ Invalid credentials: {e}")
        return False

import requests

def send_telegram(message):
    token = "7763450358:AAH32bWYyu_hXR6l-UaVMaarFGZ4YFOv6q8"
    chat_id = "6784139148"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    r = requests.post(url, data=payload)
    return r.json()
import requests

def send_telegram(message):
    token = "YOUR_TELEGRAM_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    r = requests.post(url, data=payload)
    return r.json()
    
def log_scan_to_sheet(df):
    gc = gspread.service_account(filename="/root/falah-credentials.json")
    sh = gc.open("FalahSheet")
    ws = sh.worksheet("ScanLog")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        [now, row["Symbol"], row["CMP"], row["Score"], row["Reasons"]]
        for _, row in df.iterrows()
    ]
    ws.append_rows(rows, value_input_option="RAW")
