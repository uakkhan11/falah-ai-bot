# smart_scanner.py – Multi-Timeframe, Multi-Threaded Halal Scanner

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
HALAL_LIST_PATH = "/root/falah-ai-bot/data/halal_symbols.csv"
results_lock = threading.Lock()
scan_results = []

# ---- Helper to fetch OHLC ----
def fetch_ohlc(symbol):
    try:
        end = datetime.now(IST)
        start_d = end - timedelta(days=30)
        start_15 = end - timedelta(days=5)

        daily = kite.historical_data(
            kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"],
            start_d, end, "day"
        )
        tf15 = kite.historical_data(
            kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"],
            start_15, end, "15minute"
        )

        if not daily or not tf15:
            return

        # Convert to DataFrames
        df_d = pd.DataFrame(daily)
        df_15 = pd.DataFrame(tf15)

        # Run strategies
        breakout = detect_breakout(df_d)
        rsi_ema = detect_rsi_ema_signals(df_d)
        green3 = detect_3green_days(df_d)
        darvas = detect_darvas_box(df_d)

        score = sum([breakout, rsi_ema, green3, darvas])

        if score >= 2:
            with results_lock:
                scan_results.append({
                    "Symbol": symbol,
                    "Breakout": breakout,
                    "RSI_EMA": rsi_ema,
                    "3Green": green3,
                    "Darvas": darvas,
                    "Score": score
                })
    except Exception as e:
        print(f"❌ {symbol} failed: {e}")

# ---- Scanner Executor ----
def run_scanner():
    symbols = get_halal_list(HALAL_LIST_PATH)
    threads = []

    print(f"🔍 Scanning {len(symbols)} halal stocks...")

    for sym in symbols:
        t = threading.Thread(target=fetch_ohlc, args=(sym,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"✅ Scan complete. {len(scan_results)} stocks passed filters.")
    df = pd.DataFrame(scan_results)
    df = df.sort_values(by="Score", ascending=False)
    df.to_csv("/root/falah-ai-bot/scan_results.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    run_scanner()
