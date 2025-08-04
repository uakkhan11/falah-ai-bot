# intraday_scanner.py

import pandas as pd
import os
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import joblib

model = joblib.load("model.pkl")

def calculate_features(df):
    df = df.copy()
    df['EMA10'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_signal'] = (df['close'] < df['bb_lower']) & (df['RSI'] > 40)
    return df

def apply_ai_model(df):
    features = ['RSI', 'EMA10', 'EMA21']
    df = df.dropna()
    X = df[features]
    df['ai_score'] = model.predict_proba(X)[:, 1]
    return df

def scan_intraday_folder(folder):
    results = []
    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue
        symbol = file.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(folder, file))
            if df.shape[0] < 30:
                continue
            df.columns = [c.lower() for c in df.columns]
            df = calculate_features(df)
            df = apply_ai_model(df)
            row = df.iloc[-1]
            if (
                35 < row['RSI'] < 65 and
                row['EMA10'] > row['EMA21'] and
                row['ai_score'] > 0.25
            ) or row['bb_signal']:
                results.append({
                    'symbol': symbol,
                    'RSI': round(row['RSI'], 2),
                    'ai_score': round(row['ai_score'], 3),
                    'close': row['close'],
                    'bb_signal': bool(row['bb_signal'])
                })
        except Exception:
            continue
    return pd.DataFrame(results)

if __name__ == "__main__":
    folder = "intraday_data/"
    df = scan_intraday_folder(folder)
    df.to_csv("intraday_screening_results.csv", index=False)
    print(df)
