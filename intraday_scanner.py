# intraday_scanner.py

import os
import json
import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import joblib

# === CONFIG ===
MODEL_FILE = "model.pkl"
FILTERED_FILE = "final_screened.json"  # Halal + Large/Mid Cap symbols
DATA_FOLDER = "intraday_data/"
OUTPUT_FILE = "intraday_screening_results.csv"

# Load AI Model
model = joblib.load(MODEL_FILE)

def calculate_features(df):
    """Calculate required features for the AI model."""
    df = df.copy()
    df['EMA10'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['ADX'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['VolumeChange'] = df['volume'].pct_change().fillna(0)
    return df

def apply_ai_model(df):
    """Apply AI model ensuring feature order matches training."""
    features = ['RSI', 'ADX', 'ATR', 'EMA10', 'EMA21', 'VolumeChange']
    df = df.dropna(subset=features)
    X = df[features]
    df['ai_score'] = model.predict_proba(X)[:, 1]
    return df

def scan_intraday_folder():
    """Scan intraday data for trade candidates."""
    results = []

    # Load only Halal + Large/Mid Cap symbols
    if not os.path.exists(FILTERED_FILE):
        print(f"❌ Missing filtered file: {FILTERED_FILE}")
        return pd.DataFrame()

    with open(FILTERED_FILE) as f:
        screened_data = json.load(f)
    # Handle both list and dict formats
    if isinstance(screened_data, dict):
        allowed_symbols = set(screened_data.keys())
    elif isinstance(screened_data, list):
        allowed_symbols = set(screened_data)
    else:
        print(f"❌ Unsupported format in {FILTERED_FILE}")
        return pd.DataFrame()

    for file in os.listdir(DATA_FOLDER):
        if not file.endswith(".csv"):
            continue
        symbol = file.replace(".csv", "")
        if symbol not in allowed_symbols:
            print(f"⏭ Skipping {symbol} (Not in Halal + L/M Cap list)")
            continue

        try:
            df = pd.read_csv(os.path.join(DATA_FOLDER, file))
            if df.shape[0] < 30:
                print(f"⚠️ {symbol} skipped: Not enough candles")
                continue

            df.columns = [c.lower() for c in df.columns]
            df = calculate_features(df)
            df = apply_ai_model(df)

            if df.empty:
                print(f"⚠️ {symbol} skipped: Missing required features")
                continue

            row = df.iloc[-1]
            reason = None

            # Trade condition
            if not (35 < row['RSI'] < 65):
                reason = f"RSI out of range ({row['RSI']:.2f})"
            elif not (row['EMA10'] > row['EMA21']):
                reason = "EMA10 not above EMA21"
            elif not (row['ai_score'] > 0.25):
                reason = f"AI score too low ({row['ai_score']:.3f})"

            if reason:
                print(f"⏭ {symbol} skipped: {reason}")
                continue

            results.append({
                'symbol': symbol,
                'RSI': round(row['RSI'], 2),
                'ADX': round(row['ADX'], 2),
                'ATR': round(row['ATR'], 2),
                'EMA10': round(row['EMA10'], 2),
                'EMA21': round(row['EMA21'], 2),
                'VolumeChange': round(row['VolumeChange'], 3),
                'ai_score': round(row['ai_score'], 3),
                'close': row['close']
            })
            print(f"✅ {symbol} passed scan.")

        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = scan_intraday_folder()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n📊 Scan complete. {len(df)} symbols found.")
    print(df)
