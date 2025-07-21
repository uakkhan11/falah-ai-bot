# generate_training_data.py
import pandas as pd
import os
import joblib
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = '/root/falah-ai-bot/historical_data/'
OUTPUT_FILE = 'your_training_data.csv'
TARGET_MOVE = 1.05  # 5% move within 10 candles

def process_file(filepath):
    df = pd.read_csv(filepath)
    if len(df) < 50 or 'close' not in df.columns:
        return None
    
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['ADX'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    df['EMA10'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['VolumeChange'] = df['volume'].pct_change().fillna(0)

    df['Future_High'] = df['close'].rolling(window=10, min_periods=1).max().shift(-1)
    df['Outcome'] = (df['Future_High'] >= df['close'] * TARGET_MOVE).astype(int)

    df = df.dropna(subset=['RSI', 'ATR', 'ADX', 'EMA10', 'EMA21', 'VolumeChange', 'Outcome'])
    return df[['date', 'close', 'RSI', 'ATR', 'ADX', 'EMA10', 'EMA21', 'VolumeChange', 'Outcome']]

# ✅ Main aggregation
all_data = []
for file in os.listdir(DATA_DIR):
    if not file.endswith('.csv'):
        continue
    symbol = file.replace('.csv', '')
    filepath = os.path.join(DATA_DIR, file)
    df = process_file(filepath)
    if df is not None:
        all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Final dataset saved: {OUTPUT_FILE} | Total rows={len(final_df)}")

joblib.dump(model, "model.pkl")
print("\n✅ Model trained and saved to model.pkl")
