# ws_live_prices.py

import subprocess, json

def start_all_websockets():
    # Define your tokens here (replace with your actual token list or fetch dynamically)
    token_list = [738561, 5633, 341249]  # Example
    tokens_path = "/root/falah-ai-bot/ws_tokens.json"
    
    with open(tokens_path, "w") as f:
        json.dump({"tokens": token_list}, f)

    # Start the worker
    subprocess.Popen(
        ["nohup", "python3", "ws_worker.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
