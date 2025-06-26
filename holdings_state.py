import json
import os

def load_previous_exits(exit_log_file="/root/falah-ai-bot/exited_stocks.json"):
    if not os.path.exists(exit_log_file):
        return []
    with open(exit_log_file, "r") as f:
        return json.load(f)

def update_exit_log(exit_log_file, symbol):
    exited = load_previous_exits(exit_log_file)
    if symbol not in exited:
        exited.append(symbol)
        with open(exit_log_file, "w") as f:
            json.dump(exited, f)
