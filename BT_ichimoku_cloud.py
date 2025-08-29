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

def ichimoku(df):
    df = df.copy()
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2
    
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2
    
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    
    df['chikou_span'] = df['close'].shift(-26)
    return df

def generate_signals(df):
    df = df.copy()
    # Price above Cloud
    df['price_above_cloud'] = df['close'] > df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
    # Tenkan crossing Kijun from below (bullish cross)
    df['tenkan_kijun_cross'] = (df['tenkan_sen'] > df['kijun_sen']) & (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1))
    # Buy signal when both conditions met
    df['buy_signal'] = df['price_above_cloud'] & df['tenkan_kijun_cross']
    return df

def backtest_ichimoku(df):
    df = ichimoku(df)
    df = generate_signals(df)

    trades = []
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    # Risk-reward ratio (adjust if needed)
    rr_ratio = 1.5
    
    for i in range(26, len(df)):  # start at 26 due to shifting
        if position is None:
            if df['buy_signal'].iloc[i]:
                entry_price = df['close'].iloc[i]
                stop_loss = df['low'].iloc[i-1]  # previous candle low as stop loss
                risk = entry_price - stop_loss
                if risk <= 0:
                    continue
                take_profit = entry_price + risk * rr_ratio
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
        print("No data files found in the directory.")
    else:
        df = pd.read_csv(os.path.join(DATA_PATHS['15minute'], symbols[0] + '.csv'))
        df['date'] = pd.to_datetime(df['date'])
        trades, df_with_indicators = backtest_ichimoku(df)
        summary = performance_summary(trades, df)
        print("Ichimoku Strategy Performance Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
