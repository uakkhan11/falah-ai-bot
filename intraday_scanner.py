# intraday_scanner.py
import pandas as pd
import os
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import joblib

model = joblib.load("model.pkl")

def calculate_features(df):
    df = df.copy()

    # Indicators
    df['EMA10'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['ADX'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()

    # Volume Change (%)
    df['VolumeChange'] = df['volume'].pct_change()

    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_signal'] = (df['close'] < df['bb_lower']) & (df['RSI'] > 40)

    return df

def apply_ai_model(df):
    features = ['RSI', 'ADX', 'ATR', 'EMA10', 'EMA21', 'VolumeChange']
    df = df.dropna(subset=features)
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
                print(f"⚠️ Skipping {symbol} (not enough candles)")
                continue
            df.columns = [c.lower() for c in df.columns]
            df = calculate_features(df)
            df = apply_ai_model(df)
            row = df.iloc[-1]

            # Main AI + Indicator logic
            if (
                35 < row['RSI'] < 65 and
                row['EMA10'] > row['EMA21'] and
                row['ai_score'] > 0.25
            ) or row['bb_signal']:
                results.append({
                    'symbol': symbol,
                    'RSI': round(row['RSI'], 2),
                    'ADX': round(row['ADX'], 2),
                    'ATR': round(row['ATR'], 2),
                    'ai_score': round(row['ai_score'], 3),
                    'close': round(row['close'], 2),
                    'bb_signal': bool(row['bb_signal'])
                })

        except Exception as e:
            print(f"❌ {symbol} failed: {e}")
            continue

    return pd.DataFrame(results)

if __name__ == "__main__":
    folder = "intraday_data/"
    df = scan_intraday_folder(folder)
    df.to_csv("intraday_screening_results.csv", index=False)
    print(df)
