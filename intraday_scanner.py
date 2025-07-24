# intraday_scanner.py

import os
import pandas as pd
import joblib

from indicators import intraday_vwap_bounce_strategy, intraday_rsi_breakout_strategy
from ai_engine import extract_features  # your feature function
from credentials import load_secrets

INTRADAY_DIR = "/root/falah-ai-bot/intraday_data/"
model = joblib.load("model.pkl")

def run_intraday_scan():
    results = []

    for file in os.listdir(INTRADAY_DIR):
        if file.endswith(".csv"):
            symbol = file.replace(".csv", "")
            path = os.path.join(INTRADAY_DIR, file)

            try:
                df = pd.read_csv(path)
                if len(df) < 20:
                    continue

                signal = False
                strategy = ""

                if intraday_vwap_bounce_strategy(df)['Signal'].iloc[-1]:
                    signal = True
                    strategy = "VWAP Bounce"

                elif intraday_rsi_breakout_strategy(df)['Signal'].iloc[-1]:
                    signal = True
                    strategy = "RSI Breakout"

                if signal:
                    features = extract_features(df)
                    ai_score = model.predict_proba([features])[0][1]

                    if ai_score >= 0.25:
                        results.append({
                            "Symbol": symbol,
                            "Strategy": strategy,
                            "Close": df['close'].iloc[-1],
                            "AI Score": round(ai_score, 4)
                        })

            except Exception as e:
                print(f"❌ {symbol} error: {e}")

    df_results = pd.DataFrame(results)
    print(f"📊 Intraday AI Picks: {len(df_results)}")
    print(df_results)
    return df_results


if __name__ == "__main__":
    run_intraday_scan()
