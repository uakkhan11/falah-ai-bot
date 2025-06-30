import json
import os
from datetime import datetime

def load_previous_exits(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print("⚠️ exited_stocks.json invalid format, resetting.")
            return {}
        return data
    except Exception as e:
        print(f"⚠️ Failed to load exited_stocks.json: {e}")
        return {}

def update_exit_log(filepath, symbol):
    data = load_previous_exits(filepath)
    data[symbol] = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Exit log updated for {symbol}")
    except Exception as e:
        print(f"⚠️ Failed to update exited_stocks.json: {e}")
