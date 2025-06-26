import json
import os

EXIT_LOG_FILE = "/root/falah-ai-bot/exit_log.json"

def load_previous_exits(exit_log_file="/root/falah-ai-bot/exited_stocks.json"):
    if not os.path.exists(exit_log_file):
        return []
    with open(exit_log_file, "r") as f:
        return json.load(f)

def update_exit_log(exits_dict):
    with open(EXIT_LOG_FILE, "w") as f:
        json.dump(exits_dict, f, indent=2)
