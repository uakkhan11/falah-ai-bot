# utils.py
import toml

def load_credentials():
    secrets = toml.load("/root/falah-ai-bot/.streamlit/secrets.toml")
    return secrets["zerodha"]
    
import json
import os
import datetime

def load_previous_holdings(filepath='holding_state.json'):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_holdings_state(data, filepath='holding_state.json'):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def log_exit_to_sheet(sheet, row_data):
    try:
        sheet.append_row(row_data)
        return True
    except Exception as e:
        print(f"❌ Google Sheets logging failed: {e}")
        return False

def now_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

import requests

def send_telegram(message: str):
    secrets = toml.load("/root/falah-ai-bot/.streamlit/secrets.toml")
    bot_token = secrets["telegram"]["bot_token"]
    chat_id = secrets["telegram"]["chat_id"]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")
import requests

def send_telegram(message: str):
    secrets = toml.load("/root/falah-ai-bot/.streamlit/secrets.toml")
    bot_token = secrets["telegram"]["bot_token"]
    chat_id = secrets["telegram"]["chat_id"]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")

def get_cnc_holdings(kite):
    try:
        holdings = kite.holdings()
        cnc_holdings = []
        for item in holdings:
            if item["product"] == "CNC" and item["quantity"] > 0:
                cnc_holdings.append({
                    "symbol": item["tradingsymbol"],
                    "quantity": item["quantity"],
                    "average_price": item["average_price"],
                    "last_price": item["last_price"],
                    "pnl": item["pnl"]
                })
        return cnc_holdings
    except Exception as e:
        print(f"❌ Error fetching CNC holdings: {e}")
        return []

def analyze_exit_signals(symbol, current_price, buy_price):
    try:
        trailing_sl = buy_price * 0.97  # 3% below buy price
        target_price = buy_price * 1.05  # 5% profit target

        if current_price <= trailing_sl:
            return "stoploss_hit"
        elif current_price >= target_price:
            return "target_hit"
        else:
            return "hold"
    except Exception as e:
        print(f"❌ Error analyzing signals for {symbol}: {e}")
        return "hold"



