import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================
# Enhanced Strategy Configurations
# ======================
CSV_PATH = "your_training_data.csv"
MODEL_PATH = "model.pkl"

# Trading parameters (in Indian Rupees)
INITIAL_CAPITAL = 1000000  # ₹10 Lakhs starting capital
FIXED_POSITION_SIZE = 100000  # ₹1 Lakh per trade
INITIAL_STOP_LOSS_PCT = 0.05  # 5% initial stop loss
TAKE_PROFIT_PCT = 0.10  # 10% take profit target (starting with 10%)
PROFIT_TRAIL_TRIGGER = 0.06  # Start trailing after 6% profit
ATR_MULTIPLIER = 2.5  # ATR multiplier for dynamic trailing
MIN_PRICE = 50.0
MAX_PRICE = 10000.0
TRANSACTION_COST = 0.001

# ======================
# Load and Clean Data
# ======================
print("Loading data and model for enhanced strategies...")
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower() for c in df.columns]

model = joblib.load(MODEL_PATH)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.tz_localize(None)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

print(f"Loaded {len(df)} rows of data")

# Data cleaning
initial_rows = len(df)
df = df[
    (df['close'] >= MIN_PRICE) & 
    (df['close'] <= MAX_PRICE) & 
    (df['close'].notna()) &
    (np.isfinite(df['close']))
].copy()

df['price_change'] = df['close'].pct_change().abs()
df = df[df['price_change'] < 0.5].copy()

print(f"Cleaned dataset: {len(df)} rows (removed {initial_rows - len(df)} rows)")

# ======================
# Enhanced Technical Indicators
# ======================
features = ['rsi', 'atr', 'adx', 'ema10', 'ema21', 'volumechange']
available_features = [f for f in features if f in df.columns]

# Calculate ATR for dynamic trailing stops (estimate from close if OHLCV not available)
if 'atr' not in df.columns:
    df['atr'] = df['close'].rolling(14).apply(lambda x: x.std() * 1.5)

# STRATEGY 1: BOLLINGER BANDS MEAN REVERSION
bb_data = ta.bbands(df['close'], length=20, std=2)
df['bb_upper'] = bb_data['BBU_20_2.0']
df['bb_lower'] = bb_data['BBL_20_2.0']
df['bb_middle'] = bb_data['BBM_20_2.0']

# STRATEGY 2: SUPERTREND (using close price approximation)
df['hl2'] = (df['close'] + df['close'].shift(1)) / 2  # Approximation for HL2
supertrend_data = ta.supertrend(df['close'], df['close'], df['close'], length=10, multiplier=3)
if supertrend_data is not None:
    df['supertrend'] = supertrend_data['SUPERT_10_3.0']
    df['supertrend_direction'] = supertrend_data['SUPERTd_10_3.0']
else:
    df['supertrend'] = df['close']
    df['supertrend_direction'] = 1

# STRATEGY 3: DONCHIAN CHANNELS  
df['donchian_upper'] = df['close'].rolling(20).max()
df['donchian_lower'] = df['close'].rolling(20).min()
df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2

# STRATEGY 4: WILLIAMS %R (approximation using close prices)
df['williams_r'] = ((df['close'].rolling(14).max() - df['close']) / 
                   (df['close'].rolling(14).max() - df['close'].rolling(14).min())) * -100

print(f"Enhanced features available: {available_features}")
df = df.dropna(subset=available_features + ['atr', 'bb_upper']).reset_index(drop=True)

# ======================
# Strategy Signal Generation
# ======================
print("Generating enhanced strategy signals...")
X = df[available_features]

df['ml_signal'] = model.predict(X)
df['ml_probability'] = model.predict_proba(X)[:, 1]

# BOLLINGER BANDS MEAN REVERSION STRATEGY
df['bb_signal'] = 0
bb_buy_condition = (
    (df['close'] < df['bb_lower']) &  # Price below lower band
    (df['close'] > df['close'].shift(1)) &  # Price starting to recover
    (df['close'] > df['bb_middle'].shift(5))  # Still above medium-term average
)
df.loc[bb_buy_condition, 'bb_signal'] = 1

bb_sell_condition = (
    (df['close'] > df['bb_upper']) |  # Overbought
    (df['close'] < df['bb_middle'])  # Back to mean
)
df.loc[bb_sell_condition, 'bb_signal'] = -1

# SUPERTREND STRATEGY
df['supertrend_signal'] = 0
if 'supertrend_direction' in df.columns:
    df.loc[df['supertrend_direction'] == 1, 'supertrend_signal'] = 1
    df.loc[df['supertrend_direction'] == -1, 'supertrend_signal'] = -1

# DONCHIAN BREAKOUT STRATEGY
df['donchian_signal'] = 0
donchian_buy_condition = (
    (df['close'] > df['donchian_upper'].shift(1)) &  # Breakout above upper channel
    (df['close'] > df['close'].shift(5))  # Momentum confirmation
)
df.loc[donchian_buy_condition,
