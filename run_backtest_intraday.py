import os
import pandas as pd
from indicators import add_all_indicators, intraday_vwap_bounce_strategy
from datetime import datetime

DATA_PATH = "/root/falah-ai-bot/historical_data_intraday/"  # Make sure this has 15min/60min files

results = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".csv"):
        symbol = file.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(DATA_PATH, file))
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values("datetime")
            df = intraday_vwap_bounce_strategy(df)

            trades = df[df['Signal']].copy()
            for _, row in trades.iterrows():
                entry_price = row['close']
                sl = entry_price * 0.98
                target = entry_price * 1.04
                results.append({
                    'symbol': symbol,
                    'date': row['datetime'],
                    'entry_price': entry_price,
                    'sl': sl,
                    'target': target
                })

        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

bt_df = pd.DataFrame(results)
bt_df.to_csv("intraday_backtest_results.csv", index=False)
print(f"✅ Backtest completed. Total trades: {len(bt_df)}")
