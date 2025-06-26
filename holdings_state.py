import json
import os

EXIT_LOG_FILE = "/root/falah-ai-bot/exit_log.json"

def load_previous_exits():
    if os.path.exists(EXIT_LOG_FILE):
        with open(EXIT_LOG_FILE, "r") as f:
            return json.load(f)
    return {}

def update_exit_log(exits_dict):
    with open(EXIT_LOG_FILE, "w") as f:
        json.dump(exits_dict, f, indent=2)
