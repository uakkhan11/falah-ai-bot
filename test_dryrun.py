# test_dryrun.py – Dry Run Simulation of Falāh Bot

import time
import random
import gspread
import requests
from oauth2client.service_account import ServiceAccountCredentials
import toml
import datetime

# Load secrets
with open("/root/falah-ai-bot/.streamlit/secrets.toml", "r") as f:
    secrets = toml.load(f)

SHEET_KEY = secrets.get("global", {}).get("google_sheet_key")
TELEGRAM_TOKEN = secrets.get("telegram", {}).get("bot_token")
TELEGRAM_CHAT_ID = secrets.get("telegram", {}).get("chat_id")
CREDS_JSON = "falah-credentials.json"

# Init Google Sheet
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_JSON, scope)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_KEY)

# Dummy stock simulation
dummy_stocks = [
    {"Symbol": "TATAMOTORS", "CMP": 920, "AI Score": 88},
    {"Symbol": "INFY", "CMP": 1480, "AI Score": 83},
    {"Symbol": "HAL", "CMP": 5240, "AI Score": 90},
    {"Symbol": "HDFCAMC", "CMP": 3850, "AI Score": 70}
]

total_capital = 100000
filtered = [s for s in dummy_stocks if s["AI Score"] >= 75]
total_score = sum(s["AI Score"] for s in filtered)

print("🔍 Simulated Candidates:")
for stock in filtered:
    weight = stock["AI Score"] / total_score
    allocation = round(weight * total_capital, 2)
    qty = int(allocation / stock["CMP"])
    stock.update({"Weight": weight, "Allocation": allocation, "Est. Qty": qty})
    print(f"{stock['Symbol']} - Score: {stock['AI Score']} | Allocated: ₹{allocation} | Qty: {qty}")

# Log to Google Sheet
ws = sheet.worksheet("LivePositions")
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
for stock in filtered:
    if stock["Est. Qty"] > 0:
        ws.append_row([
            stock["Symbol"],
            stock["Est. Qty"],
            stock["CMP"],
            timestamp
        ])

# Telegram Message
msg = f"🧪 *Falāh Bot Dry Run Complete*\n\n"
for s in filtered:
    msg += f"🔹 {s['Symbol']} | Score: {s['AI Score']} | ₹{s['Allocation']} | Qty: {s['Est. Qty']}\n"
msg += "\n📋 Logged to Google Sheet ✅"

url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
print("✅ Telegram alert sent.")
