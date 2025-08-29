import os
import pandas as pd
import numpy as np

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}

def get_symbols_from_15min_data():
    files = os.listdir(DATA_PATHS['15minute'])
    return [os.path.splitext(f)[0] for f in files if f.endswith('.csv')]

def atr(df, period=1):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def ema(series, period=100):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=12):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def supertrend(df, atr_len=1, factor=4, ema_len=100):
    df = df.copy()
    df['ATR'] = atr(df, period=atr_len)
    hl2 = (df['high'] + df['low']) / 2
    df['EMA100'] = ema(df['Close'], ema_len)
    df['Upper Basic Band'] = hl2 + (factor * df['ATR'])
    df['Lower Basic Band'] = hl2 - (factor * df['ATR'])

    df['Upper Band'] = df['Upper Basic Band']
    df['Lower Band'] = df['Lower Basic Band']

    supertrend = [True]  # True means uptrend, False downtrend
    for i in range(1, len(df)):
        curr_close = df['Close'].iloc[i]
        prev_st = supertrend[-1]
        prev_upper_band = df['Upper Band'].iloc[i-1]
        prev_lower_band = df['Lower Band'].iloc[i-1]
        curr_upper_band = df['Upper Band'].iloc[i]
        curr_lower_band = df['Lower Band'].iloc[i]

        if prev_st:
            if curr_close <= curr_lower_band:
                supertrend.append(False)
            else:
                supertrend.append(True)
                if curr_lower_band < prev_lower_band:
                    df.at[df.index[i], 'Lower Band'] = curr_lower_band
                else:
                    df.at[df.index[i], 'Lower Band'] = prev_lower_band
        else:
            if curr_close >= curr_upper_band:
                supertrend.append(True)
            else:
                supertrend.append(False)
                if curr_upper_band > prev_upper_band:
                    df.at[df.index[i], 'Upper Band'] = curr_upper_band
                else:
                    df.at[df.index[i], 'Upper Band'] = prev_upper_band

    df['Supertrend'] = supertrend
    return df

def rstmtf_histo_signal(df, period=12):
    df = df.copy()
    df['RSI'] = rsi(df['close'], period)
    df['HistoSignal'] = df['RSI'] > 50
    return df

def backtest_strategy(df):
    df = supertrend(df, atr_len=1, factor=4, ema_len=100)
    df = rstmtf_histo_signal(df, period=12)

    trades = []
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    for i in range(1, len(df)):
        if position is None:
            if df['Supertrend'].iloc[i] and df['HistoSignal'].iloc[i] and df['close'].iloc[i] > df['open'].iloc[i]:
                entry_price = df['Close'].iloc[i]
                prev_low = df['low'].iloc[:i].min() if i > 0 else df['low'].iloc[i]
                stop_loss = prev_low
                risk = entry_price - stop_loss
                if risk <= 0:
                    continue  # Skip if invalid risk (stop loss above entry)
                take_profit = entry_price + (risk * 1.5)
                position = 'long'
                trades.append({'EntryIndex': i, 'EntryPrice': entry_price, 'StopLoss': stop_loss,
                               'TakeProfit': take_profit, 'ExitIndex': None, 'ExitPrice': None, 'Result': None})

        elif position == 'long':
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]

            if low <= stop_loss:
                trades[-1]['ExitIndex'] = i
                trades[-1]['ExitPrice'] = stop_loss
                trades[-1]['Result'] = stop_loss - trades[-1]['EntryPrice']
                position = None
            elif high >= take_profit:
                trades[-1]['ExitIndex'] = i
                trades[-1]['ExitPrice'] = take_profit
                trades[-1]['Result'] = take_profit - trades[-1]['EntryPrice']
                position = None

    return trades

def performance_summary(trades, df):
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['Result'] and t['Result'] > 0)
    losing_trades = total_trades - winning_trades
    gross_profit = sum(t['Result'] for t in trades if t['Result'] and t['Result'] > 0)
    gross_loss = sum(t['Result'] for t in trades if t['Result'] and t['Result'] <= 0)
    net_profit = gross_profit + gross_loss
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_profit = (net_profit / total_trades) if total_trades > 0 else 0

    # Historical time span
    start_date = df['Date'].iloc[0]
    end_date = df['Date'].iloc[-1]
    date_range_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    years = date_range_days / 365

    summary = {
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate (%)': round(win_rate, 2),
        'Gross Profit': round(gross_profit, 2),
        'Gross Loss': round(gross_loss, 2),
        'Net Profit': round(net_profit, 2),
        'Average Profit Per Trade': round(avg_profit, 4),
        'Time Span Years': round(years, 2),
        'Start Date': start_date,
        'End Date': end_date
    }
    return summary

if __name__ == "__main__":
    import os
    import pandas as pd

    symbols = get_symbols_from_15min_data()
    if not symbols:
        print("No data files found in the directory.")
    else:
        # Load the first symbol for testing
        df = pd.read_csv(os.path.join(DATA_PATHS['15minute'], symbols[0] + '.csv'))
        # Make sure Date column is parsed as datetime
        df['Date'] = pd.to_datetime(df['date'])
        
        trades = backtest_strategy(df)
        summary = performance_summary(trades, df)
        print("Performance Summary:")
        for k, v in summary.items():
            print(f"{k}: {v}")

# Example usage:
# symbols = get_symbols_from_15min_data()
# df = pd.read_csv(os.path.join(DATA_PATHS['15minute'], symbols[0] + '.csv'))
# trades = backtest_strategy(df)
# summary = performance_summary(trades, df)
# print(summary)
