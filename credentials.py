# credentials.py

import json
import os
import requests
import gspread
from datetime import datetime
from kiteconnect import KiteConnect

SECRETS_PATH = "/root/falah-ai-bot/secrets.json"
TOKEN_PATH = "/root/falah-ai-bot/access_token.json"
CRED_PATH = "/root/falah-credentials.json"

def load_secrets(path=SECRETS_PATH):
    with open(path, "r") as f:
        return json.load(f)

def get_kite():
    """
    Load KiteConnect with access_token from file and return the Kite client.
    """
    secrets = load_secrets()
    kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])

    if not os.path.exists(TOKEN_PATH):
        raise Exception("❌ Access token file not found. Please generate it from the Streamlit dashboard first.")

    with open(TOKEN_PATH, "r") as f:
        token = json.load(f)["access_token"]

    kite.set_access_token(token)
    return kite

def validate_kite(kite):
    """
    Validate the loaded credentials by calling Kite profile.
    """
    try:
        profile = kite.profile()
        print("✅ Valid credentials loaded for:", profile["user_name"])
        return True
    except Exception as e:
        print(f"❌ Invalid Kite credentials: {e}")
        return False

def send_telegram(message):
    """
    Send a Telegram alert.
    """
    secrets = load_secrets()
    token = secrets["telegram"]["bot_token"]
    chat_id = secrets["telegram"]["chat_id"]

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    r = requests.post(url, data=payload)
    return r.json()

def log_scan_to_sheet(df):
    """
    Log scan results to a Google Sheet.
    """
    gc = gspread.service_account(filename=CRED_PATH)
    sh = gc.open("FalahSheet")
    ws = sh.worksheet("ScanLog")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        [now, row["Symbol"], row["CMP"], row["Score"], row["Reasons"]]
        for _, row in df.iterrows()
    ]
    ws.append_rows(rows, value_input_option="RAW")
