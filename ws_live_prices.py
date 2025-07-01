# ws_live_prices.py

import subprocess

def start_websockets(api_key, access_token, tokens, batch_size=300):
    """
    Starts multiple KiteTicker WebSocket connections in separate subprocesses.
    """
    batches = [tokens[i:i + batch_size] for i in range(0, len(tokens), batch_size)]
    print(f"✅ Splitting tokens into {len(batches)} batch(es).")

    for i, batch in enumerate(batches, start=1):
        token_str = ",".join(str(t) for t in batch)
        print(f"✅ Launching subprocess for batch {i} ({len(batch)} tokens)")

        subprocess.Popen(
            [
                "python3",
                "/root/falah-ai-bot/ws_worker.py",
                api_key,
                access_token,
                token_str
            ]
        )
