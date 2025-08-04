# daily_scanner.py (Improved Real-time Version)

import os
import json
import pandas as pd
import joblib
from datetime import datetime
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from fetch_historical_batch import fetch_all_historical
from live_price_reader import get_symbol_price_map

# Load AI Model
model = joblib.load("model.pkl")

HISTORICAL_DIR = "/root/falah-ai-bot/historical_data"

def calculate_features(df):
    """Add swing-trading features."""
    df['EMA10'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_signal'] = (
        (df['close'] < df['bb_lower']) &
        (df['RSI'] > 40) &
        (df['ATR'] / df['close'] < 0.03)  # volatility filter
    )
    return df

def apply_ai_model(df):
    features = ['RSI', 'ATR', 'EMA10', 'EMA21']
    df = df.dropna()
    X = df[features]
    df['ai_score'] = model.predict_proba(X)[:, 1]
    return df

def scan_daily_folder(folder):
    results = []

    print(f"📊 Starting Daily Scan at {datetime.now().strftime('%H:%M:%S')}...")
    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue
        symbol = file.replace(".csv", "")

        try:
            df = pd.read_csv(os.path.join(folder, file))
            if df.shape[0] < 30:
                print(f"⚠️ Skipped {symbol}: Not enough candles ({df.shape[0]} rows)")
                continue

            df.columns = [c.lower() for c in df.columns]

            # Ensure last candle reflects current LTP
            ltp_map = get_symbol_price_map(symbols=[symbol], force_live=True)
            if symbol in ltp_map:
                df.iloc[-1, df.columns.get_loc("close")] = ltp_map[symbol]

            df = calculate_features(df)
            df = apply_ai_model(df)

            row = df.iloc[-1]
            rsi = row['RSI']
            ema_ok = row['EMA10'] > row['EMA21']
            ai_ok = row['ai_score'] > 0.25
            bb_ok = row['bb_signal']

            print(f"🔍 {symbol} | RSI: {rsi:.2f} | EMA10>EMA21: {ema_ok} | AI: {row['ai_score']:.3f} | BB: {bb_ok}")

            if (35 < rsi < 65 and ema_ok and ai_ok) or bb_ok:
                results.append({
                    'symbol': symbol,
                    'RSI': round(rsi, 2),
                    'ai_score': round(row['ai_score'], 3),
                    'close': row['close'],
                    'ATR': round(row['ATR'], 3),
                    'bb_signal': bool(bb_ok)
                })

        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

    print(f"✅ Daily scan complete. {len(results)} symbols matched criteria.")
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Step 1: Fetch latest historical EOD data (including today’s candle if open)
    fetch_all_historical()

    # Step 2: Scan updated data
    df = scan_daily_folder(HISTORICAL_DIR)
    df.to_csv("daily_screening_results.csv", index=False)
    print(df)
