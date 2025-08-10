# --- In add_indicators(df) ---
# Existing TA calcs...
df['rsi'] = ta.rsi(df['close'], length=14)
macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
df['macd'], df['macd_signal'] = macd['MACD_12_26_9'], macd['MACDs_12_26_9']
df['macd_hist'] = macd['MACDh_12_26_9']

# VWAP
df['vwap'] = (df['volume']*(df['high']+df['low']+df['close'])/3).cumsum() / df['volume'].cumsum()

# Pivots (Classic)
pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1))/3
df['pivot'] = pivot
df['r1'] = 2*pivot - df['low'].shift(1)
df['s1'] = 2*pivot - df['high'].shift(1)

# Fibonacci retracements from recent swing high/low
lookback = 20
recent_high = df['high'].rolling(lookback).max()
recent_low  = df['low'].rolling(lookback).min()
df['fib_38'] = recent_high - (recent_high - recent_low) * 0.382
df['fib_61'] = recent_high - (recent_high - recent_low) * 0.618

# Nash placeholder
df['nash_zone'] = (recent_high + recent_low) / 2  # midpoint as placeholder

# --- In combine_signals(df, params) ---
# Include extra filter logic:
momentum_ok = (df['rsi'] >= 35) & (df['rsi'] <= 70)
macd_ok     = (df['macd'] > df['macd_signal']) & (df['macd_hist'] > 0)
vwap_ok     = df['close'] > df['vwap']
pivot_ok    = df['close'] > df['pivot']
fib_ok      = df['close'] > df['fib_38']  # adjust logic to your style
vol_spike   = df['volume'] > 1.5 * df['vol_sma20']

extra_conf  = momentum_ok & macd_ok & vwap_ok & pivot_ok & fib_ok & vol_spike

# Then combine with your existing regime + CE/ST confirms:
mask_break  = regime_break & chand_or_st & extra_conf
mask_pull   = regime_pull & chand_or_st & extra_conf

# ... set entry_signal & entry_type based on these masks
