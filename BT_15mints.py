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

def backtest_simplified_strategy(df):
    df = df.copy()
    df['ema_short'] = ema(df['close'], 12)
    df['ema_long'] = ema(df['close'], 26)
    df['rsi'] = rsi(df['close'], 14)

    stop_loss_pct = 0.005  # 0.5%
    take_profit_pct = 0.01  # 1%

    trades = []
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    for i in range(1, len(df)):
        ma_cross = df['ema_short'].iloc[i] > df['ema_long'].iloc[i] and df['ema_short'].iloc[i-1] <= df['ema_long'].iloc[i-1]
        rsi_ok = df['rsi'].iloc[i] > 30

        print(f"Index {i} - MA_Cross: {ma_cross}, RSI: {df['rsi'].iloc[i]:.2f}, Position: {position}")

        if position is None:
            if ma_cross and rsi_ok:
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
                print(f"Entered Long at index {i}, price {entry_price:.2f}")
        elif position == 'long':
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]

            if low <= stop_loss:
                trades[-1]['ExitIndex'] = i
                trades[-1]['ExitPrice'] = stop_loss
                trades[-1]['Result'] = stop_loss - trades[-1]['EntryPrice']
                position = None
                print(f"Stop Loss hit at index {i}, price {stop_loss:.2f}")
            elif high >= take_profit:
                trades[-1]['ExitIndex'] = i
                trades[-1]['ExitPrice'] = take_profit
                trades[-1]['Result'] = take_profit - trades[-1]['EntryPrice']
                position = None
                print(f"Take Profit hit at index {i}, price {take_profit:.2f}")

    return trades, df

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

if __name__ == "__main__":
    symbols = get_symbols_from_15min_data()
    if not symbols:
        print("No 15min data files found.")
    else:
        df = pd.read_csv(os.path.join(DATA_PATHS['15minute'], symbols[0] + '.csv'))
        df['date'] = pd.to_datetime(df['date'])
        trades, df_with_indicators = backtest_simplified_strategy(df)
        summary = performance_summary(trades, df)
        print("Simplified Strategy Backtest Performance Summary:")
        for k, v in summary.items():
            print(f"{k}: {v}")
