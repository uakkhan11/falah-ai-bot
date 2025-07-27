# price_fetcher.py
import os
import pandas as pd

DATA_DIR = "/root/falah-ai-bot/historical_data"

def get_price(symbol):
    """
    Fetches the latest close price from historical CSV data.
    """
    filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No historical data found for {symbol}")

    df = pd.read_csv(filepath)
    if len(df) == 0:
        raise ValueError(f"No data in {symbol}.csv")

    latest = df.iloc[-1]
    return float(latest["close"])
