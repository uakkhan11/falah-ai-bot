import pandas as pd
import pandas_ta as ta

# === Config ===
CSV_PATH = "/root/falah-ai-bot/your_training_data.csv"   # Update if needed
TARGET_COL = "outcome"                                   # Your target column name (case-insensitive)

# Load CSV and lowercase columns
df = pd.read_csv(CSV_PATH)
df.columns = [col.lower() for col in df.columns]

# Parse 'date' and sort
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.sort_values('date', inplace=True)

# Check required raw columns for indicators
required_price_cols = ['close']
for col in required_price_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' missing from data.")

# Compute missing basic indicators if absent
if 'rsi' not in df.columns:
    df['rsi'] = ta.rsi(df['close'], length=14)

if all(c in df.columns for c in ['high', 'low', 'close']):
    if 'atr' not in df.columns:
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    if 'adx' not in df.columns:
        df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['adx_14']

if 'ema10' not in df.columns:
    df['ema10'] = ta.ema(df['close'], length=10)
if 'ema21' not in df.columns:
    df['ema21'] = ta.ema(df['close'], length=21)

if 'volume' in df.columns:
    if 'volumechange' not in df.columns:
        df['volumechange'] = df['volume'].pct_change().fillna(0)
else:
    df['volumechange'] = pd.NA

# MACD (needs only close)
macd_ind = ta.macd(df['close'])
df['macd_hist'] = macd_ind['MACDh_12_26_9']
df['macd_signal'] = macd_ind['MACDs_12_26_9']

# Conditionally compute volume/price-dependent indicators
if all(c in df.columns for c in ['high', 'low', 'volume']):
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch['stochk_14_3_3']
    df['stoch_d'] = stoch['stochd_14_3_3']
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
else:
    df['stoch_k'] = pd.NA
    df['stoch_d'] = pd.NA
    df['obv'] = pd.NA
    df['vwap'] = pd.NA

# Define feature list dynamically depending on available columns
features = ['rsi', 'ema10', 'ema21', 'macd_hist', 'macd_signal']
for optional_feat in ['atr', 'adx', 'volumechange', 'stoch_k', 'stoch_d', 'obv', 'vwap']:
    if optional_feat in df.columns and pd.api.types.is_numeric_dtype(df[optional_feat]):
        features.append(optional_feat)

if TARGET_COL.lower() not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

target = TARGET_COL.lower()

# Display missing value counts before dropping
print("Missing values count per feature and target:")
print(df[features + [target]].isna().sum())

# Drop initial rows affected by indicator lookbacks (e.g., 30 days)
lookback = 40
df = df.iloc[lookback:].copy()

# Check rows after trimming
print(f"Rows after trimming first {lookback} rows: {len(df)}")

# Drop rows missing target only, keep rows where features may have NaN (optional)
df.dropna(subset=[target], inplace=True)

# Optionally, fill or interpolate missing features if needed, or drop rows with NaN in features
df.dropna(subset=features, inplace=True)

# Prepare X and y
X = df[features]
y = df[target].astype(int)

print(f"✅ Dataset ready: {len(X)} samples | Positives: {y.sum()} | Negatives: {len(y) - y.sum()}")
print(f"Features used: {features}")
