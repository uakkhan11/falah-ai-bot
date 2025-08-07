# filename: generate_model_features.py

import pandas as pd
import ta  # Technical Analysis library
import os

def generate_features(input_csv, output_csv):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file '{input_csv}' not found")

    df = pd.read_csv(input_csv)
    numeric_cols = ['close', 'high', 'low', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # ✅ Drop rows with missing or invalid numeric data
    df.dropna(subset=numeric_cols, inplace=True)

    if "close" not in df.columns:
        raise ValueError("❌ 'close' column is required for feature generation")

    # Standardize column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]

    # Add more indicators
    df['ema10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['volumechange'] = df['volume'].pct_change()

    # Drop rows with NaN
    df.dropna(inplace=True)

    df.to_csv(output_csv, index=False)
    print(f"✅ Feature-enhanced file saved to: {output_csv}")

if __name__ == "__main__":
    # Modify filenames if needed
    generate_features("model_training_data_raw.csv", "model_training_data.csv")
