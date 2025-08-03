# credentials.py

import json
import os
import requests
import gspread
from datetime import datetime
from kiteconnect import KiteConnect

# --------- CONFIG PATHS ----------
SECRETS_PATH = "/root/falah-ai-bot/secrets.json"
CRED_PATH = "/root/falah-credentials.json"

# --------- LOAD SECRETS ----------
def load_secrets(path=SECRETS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"secrets.json not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

# --------- GET KITE SESSION ----------
def get_kite():
    """
    Returns a KiteConnect client with access token loaded from secrets.json.
    This is mobile-friendly, no manual steps required.
    """
    secrets = load_secrets()
    kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])
    kite.set_access_token(secrets["zerodha"]["access_token"])
    return kite

# --------- VALIDATE KITE SESSION ----------
def validate_kite(kite):
    """
    Checks if the credentials are valid by fetching the user profile.
    """
    try:
        profile = kite.profile()
        print("✅ Valid Kite credentials for:", profile["user_name"])
        return True
    except Exception as e:
        print(f"❌ Invalid credentials: {e}")
        return False

# --------- TELEGRAM ALERT ----------
def send_telegram(message):
    """
    Sends a message to Telegram bot.
    """
    secrets = load_secrets()
    token = secrets["telegram"]["bot_token"]
    chat_id = secrets["telegram"]["chat_id"]

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    r = requests.post(url, data=payload)
    return r.json()

# --------- LOG SCAN TO GOOGLE SHEETS ----------
def log_scan_to_sheet(df):
    """
    Logs trade scan results to a Google Sheet.
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
