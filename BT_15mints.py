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

# Technical Indicators
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def supertrend(df, atr_len=7, factor=3):
    df = df.copy()
    df['ATR'] = atr(df, period=atr_len)
    hl2 = (df['high'] + df['low']) / 2
    df['Upper Basic Band'] = hl2 + (factor * df['ATR'])
    df['Lower Basic Band'] = hl2 - (factor * df['ATR'])
    df['Upper Band'] = df['Upper Basic Band']
    df['Lower Band'] = df['Lower Basic Band']

    supertrend = [True]  # True means uptrend, False downtrend
    for i in range(1, len(df)):
        curr_close = df['close'].iloc[i]
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

# Candlestick Bullish Engulfing Pattern
def bullish_engulfing(df):
    df = df.copy()
    df['bullish_engulf'] = False
    for i in range(1, len(df)):
        prev_open = df['open'].iloc[i-1]
        prev_close = df['close'].iloc[i-1]
        curr_open = df['open'].iloc[i]
        curr_close = df['close'].iloc[i]

        cond1 = (prev_close < prev_open)  # Previous candle bearish
        cond2 = (curr_close > curr_open)  # Current candle bullish
        cond3 = (curr_close > prev_open) and (curr_open < prev_close)  # Engulfing body
        if cond1 and cond2 and cond3:
            df.at[df.index[i], 'bullish_engulf'] = True
    return df

# Backtest Logic with combined signals and risk mgmt
def backtest_strategy(df):
    df = df.copy()
    
    # Calculate indicators
    df['ema_short'] = ema(df['close'], 12)
    df['ema_long'] = ema(df['close'], 26)
    df['rsi'] = rsi(df['close'], 14)
    df['macd_line'], df['signal_line'], df['macd_hist'] = macd(df['close'])
    df = supertrend(df, atr_len=7, factor=3)
    df = bullish_engulfing(df)

    # Risk management
    stop_loss_pct = 0.005  # 0.5% stop loss
    take_profit_pct = 0.01  # 1% take profit

    trades = []
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    for i in range(1, len(df)):
        if position is None:
            # Entry conditions (all must be True)
            ma_cross = df['ema_short'].iloc[i] > df['ema_long'].iloc[i] and df['ema_short'].iloc[i-1] <= df['ema_long'].iloc[i-1]
            rsi_ok = df['rsi'].iloc[i] > 30  # filter out oversold
            macd_ok = df['macd_hist'].iloc[i] > 0
            supertrend_ok = df['Supertrend'].iloc[i] == True
            bullish_candle = df['close'].iloc[i] > df['open'].iloc[i]
            engulf_ok = df['bullish_engulf'].iloc[i]
            volume_ok = df['volume'].iloc[i] > df['volume'].rolling(window=20).mean().iloc[i]

            if all([ma_cross, rsi_ok, macd_ok, supertrend_ok, bullish_candle, engulf_ok, volume_ok]):
                entry_price = df['close'].iloc[i]
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                position = 'long'
                trades.append({'EntryIndex': i,
                               'EntryPrice': entry_price,
                               'StopLoss': stop_loss,
                               'TakeProfit': take_profit,
                               'ExitIndex': None,
                               'ExitPrice': None,
                               'Result': None})

        elif position == 'long':
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]

            # Check stop loss hit
            if low <= stop_loss:
                trades[-1]['ExitIndex'] = i
                trades[-1]['ExitPrice'] = stop_loss
                trades[-1]['Result'] = stop_loss - trades[-1]['EntryPrice']
                position = None
            # Check take profit hit
            elif high >= take_profit:
                trades[-1]['ExitIndex'] = i
                trades[-1]['ExitPrice'] = take_profit
                trades[-1]['Result'] = take_profit - trades[-1]['EntryPrice']
                position = None

    return trades, df

# Performance Summary
def performance_summary(trades, df):
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['Result'] and t['Result'] > 0)
    losing_trades = total_trades - winning_trades
    gross_profit = sum(t['Result'] for t in trades if t['Result'] and t['Result'] > 0)
    gross_loss = sum(t['Result'] for t in trades if t['Result'] and t['Result'] <= 0)
    net_profit = gross_profit + gross_loss
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_profit = (net_profit / total_trades) if total_trades > 0 else 0

    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
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

def save_summary_to_file(summary, filename="backtest_results.txt"):
    with open(filename, "w") as f:
        f.write("Combined Strategy Backtest Performance Summary:\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    symbols = get_symbols_from_15min_data()
    if not symbols:
        print("No 15min data files found.")
    else:
        # Load first file
        df = pd.read_csv(os.path.join(DATA_PATHS['15minute'], symbols[0] + '.csv'))
        df['date'] = pd.to_datetime(df['date'])

        trades, df_with_indicators = backtest_strategy(df)
        summary = performance_summary(trades, df)

        # Print to console
        print("Combined Strategy Backtest Performance Summary:")
        for k, v in summary.items():
            print(f"{k}: {v}")

        # Save detailed summary to file
        save_summary_to_file(summary)
        print(f"Detailed summary saved to backtest_results.txt")
