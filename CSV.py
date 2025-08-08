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

# Compute missing indicators/features
if "rsi" not in df.columns:
    df["rsi"] = ta.rsi(df["close"], length=14)

if all(col in df.columns for col in ['high', 'low', 'close']):
    if "atr" not in df.columns:
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    if "adx" not in df.columns:
        df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=14)["adx_14"]

if "ema10" not in df.columns:
    df["ema10"] = ta.ema(df["close"], length=10)
if "ema21" not in df.columns:
    df["ema21"] = ta.ema(df["close"], length=21)

if "volume" in df.columns:
    if "volumechange" not in df.columns:
        df["volumechange"] = df["volume"].pct_change().fillna(0)
else:
    df["volumechange"] = pd.NA

# MACD calculation
macd = ta.macd(df['close'])
df['macd_hist'] = macd['macdh_12_26_9']
df['macd_signal'] = macd['macds_12_26_9']

if all(col in df.columns for col in ['high', 'low', 'volume']):
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

# Build features list dynamically based on available columns
features = ["rsi", "ema10", "ema21", "macd_hist", "macd_signal"]

if 'atr' in df.columns and 'adx' in df.columns:
    features += ["atr", "adx"]
if 'volumechange' in df.columns and pd.api.types.is_numeric_dtype(df['volumechange']):
    features.append('volumechange')
if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
    features += ['stoch_k', 'stoch_d']
if 'obv' in df.columns:
    features.append('obv')
if 'vwap' in df.columns:
    features.append('vwap')

target = "outcome"
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset.")

print(f"Rows before dropna: {len(df)}")
df.dropna(subset=features + [target], inplace=True)
print(f"Rows after dropna: {len(df)}")

X = df[features]
y = df[target].astype(int)

print(f"✅ Dataset ready: {len(X)} samples | Positives: {y.sum()} | Negatives: {len(y) - y.sum()}")
