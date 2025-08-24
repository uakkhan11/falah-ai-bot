import pandas as pd
from strategy_utils import (
    add_indicators, breakout_signal,
    bb_breakout_signal, bb_pullback_signal
)

def load_sample_data(symbol):
    file_path = f"swing_data/{symbol}.csv"
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def compute_signals(df):
    df = add_indicators(df)
    df = breakout_signal(df)
    df = bb_breakout_signal(df)
    df = bb_pullback_signal(df)
    return df

def debug_bollinger_conditions(df):
    print("\n--- Latest 20 rows Bollinger Band and Volume data ---")
    print(df[['date', 'close', 'bb_upper', 'bb_lower', 'volume', 'vol_sma20']].tail(20))
    
    print("\n--- Calculated Bollinger Band Signals ---")
    print(df[['date', 'bb_breakout_signal', 'bb_pullback_signal']].tail(20))
    
    print("\n--- Bollinger breakout condition check ---")
    condition_breakout = (df['close'] > df['bb_upper']) & (df['volume'] > 1.1 * df['vol_sma20'])
    print(condition_breakout.tail(20))
    
    print("\n--- Bollinger pullback condition check ---")
    condition_pullback = (df['close'].shift(1) < df['bb_lower'].shift(1)) & (df['close'] > df['bb_lower'])
    print(condition_pullback.tail(20))

def main():
    symbol = "RELIANCE"
    df = load_sample_data(symbol)
    df = compute_signals(df)
    debug_bollinger_conditions(df)

if __name__ == "__main__":
    main()
