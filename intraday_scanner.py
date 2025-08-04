# intraday_scanner.py (Improved Real-time Version)

import os
import pandas as pd
import joblib
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime

from fetch_intraday_data import fetch_intraday_data
from live_price_reader import get_symbol_price_map

# Load AI Model
model = joblib.load("model.pkl")

INTRADAY_DIR = "/root/falah-ai-bot/intraday_data"

def calculate_features(df):
    """Add all indicators used in trade logic"""
    df['EMA10'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    # More strict BB signal: price below lower band, RSI > 40, low ATR noise
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

def scan_intraday_folder(folder):
    results = []

    print(f"📊 Starting Intraday Scan at {datetime.now().strftime('%H:%M:%S')}...")
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

            # Ensure we have the latest LTP for the last candle
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

            # Debug log for each stock
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

    print(f"✅ Intraday scan complete. {len(results)} symbols matched criteria.")
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Step 1: Fetch fresh data
    try:
        with open("/root/falah-ai-bot/symbol_to_token.json") as f:
            all_symbols = list(json.load(f).keys())
    except:
        all_symbols = []
        print("⚠️ No symbol list found, scanning whatever is in intraday_data")

    fetch_intraday_data(all_symbols)

    # Step 2: Scan freshly fetched data
    df = scan_intraday_folder(INTRADAY_DIR)
    df.to_csv("intraday_screening_results.csv", index=False)
    print(df)
