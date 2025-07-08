import json
import requests

def send_telegram(message):
    with open("secrets.json") as f:
        secrets = json.load(f)
    BOT_TOKEN = secrets["telegram"]["bot_token"]
    CHAT_ID = secrets["telegram"]["chat_id"]

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    requests.post(url, data=payload)
