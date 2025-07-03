# smart_scanner.py – Multi-Timeframe, Multi-Threaded Halal Scanner
from utils import get_halal_list

import threading
import pandas as pd
import pytz
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from utils import load_credentials, get_halal_list
from indicators import detect_breakout, detect_rsi_ema_signals, detect_3green_days, detect_darvas_box

# ---- Init ----
IST = pytz.timezone('Asia/Kolkata')
secrets = load_credentials()
kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])
kite.set_access_token(secrets["zerodha"]["access_token"])

# ---- Config ----
HALAL_LIST_PATH = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"
results_lock = threading.Lock()
scan_results = []

def run_smart_scan():
    """
    Runs the scan and returns a DataFrame of results.
    """
    # Clear previous results
    global scan_results
    scan_results = []

    # Load halal symbols
    halal_symbols = get_halal_list(HALAL_LIST_PATH)

    # Split symbols into chunks for threads
    chunks = [halal_symbols[i::5] for i in range(5)]  # 5 threads

    def worker(symbols_chunk):
        for symbol in symbols_chunk:
            try:
                # Example pseudo-logic; replace with your actual indicator calls
                has_breakout = detect_breakout(kite, symbol)
                has_rsi = detect_rsi_ema_signals(kite, symbol)
                has_3green = detect_3green_days(kite, symbol)
                has_darvas = detect_darvas_box(kite, symbol)

                score = sum([has_breakout, has_rsi, has_3green, has_darvas])

                if score >= 2:
                    cmp = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["last_price"]

                    with results_lock:
                        scan_results.append({
                            "Symbol": symbol,
                            "Confidence": score / 4,
                            "CMP": cmp
                        })
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")

    # Launch threads
    threads = [threading.Thread(target=worker, args=(chunk,)) for chunk in chunks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Convert results to DataFrame
    df = pd.DataFrame(scan_results)
    df = df.sort_values(by="Confidence", ascending=False).reset_index(drop=True)
    return df

