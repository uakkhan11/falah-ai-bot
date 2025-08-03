# ws_live_prices.py
import os, subprocess, json

tokens_dir = "/root/falah-ai-bot/ws_tokens"
os.makedirs(tokens_dir, exist_ok=True)
def start_all_websockets():
    tokens_dir = "/root/falah-ai-bot/ws_tokens"
    for file in sorted(os.listdir(tokens_dir)):
        if file.endswith(".json"):
            token_path = os.path.join(tokens_dir, file)
            subprocess.Popen([
                "python3", "ws_worker.py", token_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    start_all_websockets()
