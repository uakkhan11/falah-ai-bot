import json
import os

def load_previous_exits(exit_log_file="/root/falah-ai-bot/exited_stocks.json"):
    if not os.path.exists(exit_log_file):
        return []
    with open(exit_log_file, "r") as f:
        return json.load(f)

import json
from datetime import datetime

def update_exit_log(filepath, symbol):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except:
        data = {}

    data[symbol] = datetime.now().strftime("%Y-%m-%d")

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
