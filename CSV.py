import pandas as pd
import pandas_ta as ta

# File path to your CSV
CSV_PATH = "/root/falah-ai-bot/your_training_data.csv"

# Load the data
df = pd.read_csv(CSV_PATH)

# Standardize column names to lowercase
df.columns = [col.lower() for col in df.columns]

# Parse 'date' column to datetime if exists
if "date" in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.sort_values('date', inplace=True)

# Verify required columns
required_cols = ['close']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in dataset.")

# Compute additional indicators if possible
# MACD requires only 'close'
macd = ta.macd(df['close'])
df['macd_hist'] = macd['MACDh_12_26_9']
df['macd_signal'] = macd['MACDs_12_26_9']

# Calculate other indicators only if you have 'high', 'low', 'volume'
if all(col in df.columns for col in ['high', 'low', 'volume']):
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
else:
    # If missing columns, set these features to NaN or skip
    df['stoch_k'] = pd.NA
    df['stoch_d'] = pd.NA
    df['obv'] = pd.NA
    df['vwap'] = pd.NA

# Define features and target matching your CSV structure
features = ["rsi", "atr", "adx", "ema10", "ema21", "volumechange",
            "macd_hist", "macd_signal", "stoch_k", "stoch_d", "obv", "vwap"]
target = "outcome"

# Drop rows with any missing feature or target values
df.dropna(subset=features + [target], inplace=True)

# Prepare X and y for modeling
X = df[features]
y = df[target].astype(int)

print(f"✅ Dataset ready: {len(X)} samples | Positives: {y.sum()} | Negatives: {len(y) - y.sum()}")
