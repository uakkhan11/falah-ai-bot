import pandas as pd
from strategy_utils import add_indicators, breakout_signal, bb_breakout_signal, bb_pullback_signal, combine_signals

def load_data(symbol):
    file_path = f"swing_data/{symbol}.csv"
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def main():
    symbols = ["RELIANCE", "SUNPHARMA"]  # Your test symbols

    for symbol in symbols:
        print(f"\nProcessing {symbol}")
        df = load_data(symbol)
        df = add_indicators(df)
        df = breakout_signal(df)
        df = bb_breakout_signal(df)
        df = bb_pullback_signal(df)
        df = combine_signals(df)

        print(df[['date', 'close', 'entry_signal', 'entry_type', 'breakout_signal',
                  'bb_breakout_signal', 'bb_pullback_signal']].tail(20))

if __name__ == "__main__":
    main()
