import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/path/to/your/data"

DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    'hourly': os.path.join(BASE_DIR, "intraday_data"),
    'fifteen_min': os.path.join(BASE_DIR, "scalping_data")
}

# Trading cost params (Zerodha CNC approx)
CHARGES = {
    'brokerage_per_trade': 0.0,
    'stt_buy': 0.001,       # 0.1%
    'stt_sell': 0.001,      # 0.1%
    'stamp_duty_buy': 0.00015, # 0.015%
    'dp_charges_sell': 13.5,
    'exchange_charges_rate': 0.0000345,
    'gst_rate': 0.18,
    'sebi_rate': 0.000001
}

def calculate_charges(buy_price, sell_price, qty):
    buy_value = buy_price * qty
    sell_value = sell_price * qty
    stt = buy_value*CHARGES['stt_buy'] + sell_value*CHARGES['stt_sell']
    stamp = buy_value * CHARGES['stamp_duty_buy']
    exch = (buy_value + sell_value)*CHARGES['exchange_charges_rate']
    gst = exch * CHARGES['gst_rate']
    sebi = (buy_value + sell_value)*CHARGES['sebi_rate']
    dp = CHARGES['dp_charges_sell']
    total = stt + stamp + exch + gst + sebi + dp
    return total

def load_and_prepare_data(filepath, timeframe):
    df = pd.read_csv(filepath, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Compute common indicators here depending on timeframe
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema21'] = ta.ema(df['close'], length=21)
    # Williams %R indicator example
    highest_high = df['high'].rolling(14).max()
    lowest_low = df['low'].rolling(14).min()
    df['williams_r'] = (highest_high - df['close']) / (highest_high - lowest_low) * -100
    # etc. Add other indicators as needed
    return df

def generate_ml_signals(df, model, feature_cols):
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    X = df[feature_cols]
    df.loc[X.index, 'ml_signal'] = model.predict(X)
    df.loc[X.index, 'ml_prob'] = model.predict_proba(X)[:,1]
    return df

def simple_williams_r_signal(df, oversold=-80, overbought=-20):
    df['signal'] = 0
    buy = (df['williams_r'] < oversold) & (df['williams_r'].shift(1) < oversold)
    sell = (df['williams_r'] > overbought)
    df.loc[buy, 'signal'] = 1
    df.loc[sell, 'signal'] = 0
    return df

def backtest_cnc(df, signal_col='signal', max_trades=200, take_profit=0.12, stop_loss=0.05,
                 transaction_cost_rate=0.001, initial_capital=1_000_000, position_size=100_000,
                 position_limit=5, allow_early_exit=True):
    cash = initial_capital
    positions = {}
    results = []
    trade_count = 0

    for i in range(1,len(df)):
        row = df.iloc[i]
        date = row['date']
        price = row['close']
        signal = row[signal_col]

        # Exit conditions for open positions
        to_close = []
        for pid, pos in positions.items():
            days_held = (date - pos['entry_date']).days
            ret = (price - pos['entry_price']) / pos['entry_price']
            if allow_early_exit:  
                # Exit triggered by signal flip or targets/stop-loss
                exit_due = (signal==0 or ret>=take_profit or ret<=-stop_loss or days_held >= 20)
            else:
                # Force exit only after min 1 day hold
                exit_due = (days_held >= 1 and (signal==0 or ret>=take_profit or ret<=-stop_loss or days_held >= 20))

            if exit_due:
                qty = pos['shares']
                charges = calculate_charges(pos['entry_price'], price, qty)
                sell_value = qty*price*(1-transaction_cost_rate)
                pnl = sell_value - position_size - charges
                cash += sell_value
                results.append(dict(entry_date=pos['entry_date'], exit_date=date,
                                    entry_price=pos['entry_price'], exit_price=price,
                                    pnl=pnl, ret_pct=ret*100, charges=charges))
                to_close.append(pid)
                trade_count += 1
                if trade_count >= max_trades:
                    break
        for pid in to_close:
            positions.pop(pid)
        if trade_count >= max_trades:
            break

        # Entry conditions
        if signal == 1 and len(positions) < position_limit and cash >= position_size:
            shares = position_size / price
            cost = position_size * (1 + transaction_cost_rate)
            if cash >= cost:
                positions[len(positions)+1] = dict(entry_date=date, entry_price=price, shares=shares)
                cash -= cost

    # Close remaining positions at last row
    last_date = df.iloc[-1]['date']
    last_close = df.iloc[-1]['close']
    for pid, pos in positions.items():
        days_held = (last_date - pos['entry_date']).days
        ret = (last_close - pos['entry_price']) / pos['entry_price']
        if days_held < 1 and not allow_early_exit:
            continue  # force hold for min 1 day
        qty = pos['shares']
        charges = calculate_charges(pos['entry_price'], last_close, qty)
        sell_value = qty*last_close*(1 - transaction_cost_rate)
        pnl = sell_value - position_size - charges
        cash += sell_value
        results.append(dict(entry_date=pos['entry_date'], exit_date=last_date,
                            entry_price=pos['entry_price'], exit_price=last_close,
                            pnl=pnl, ret_pct=ret*100, charges=charges))

    trades_df = pd.DataFrame(results)
    total_return = (cash - initial_capital) / initial_capital * 100
    win_rate = (trades_df['pnl'] > 0).mean() * 100 if not trades_df.empty else 0
    avg_hold = trades_df.assign(hold=(trades_df.exit_date - trades_df.entry_date).dt.days)['hold'].mean() if not trades_df.empty else 0
    sharpe = trades_df.ret_pct.mean() / trades_df.ret_pct.std() if not trades_df.empty else 0

    return trades_df, total_return, win_rate, avg_hold, sharpe

def run_full_retest():
    # Load ML model
    model = joblib.load(MODEL_PATH)
    feature_cols = ['rsi','atr','adx','ema10','ema21','volumechange']

    # For results store
    final_results = []

    # Daily timeframe test (mostly swing)
    daily_dir = DATA_PATHS['daily']
    for file in os.listdir(daily_dir):
        if not file.endswith('.csv'):
            continue
        path = os.path.join(daily_dir, file)
        df = load_and_prepare_data(path, 'daily')
        # Calculate volumechange
        df['volumechange'] = df['volume'].pct_change().fillna(0)
        # Merge ML predictions
        df_ml_raw = pd.read_csv(CSV_PATH, parse_dates=['date'])
        df_ml_raw['date'] = pd.to_datetime(df_ml_raw['date']).dt.tz_localize(None)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df_full = df.merge(df_ml_raw[['date']], on='date', how='inner')
        df_full = generate_ml_signals(df_full, model, feature_cols)
        # Backtest ML CNC, early-exit allowed
        trades, ret, win, hold, sharpe = backtest_cnc(df_full, 'ml_signal', allow_early_exit=True)
        final_results.append({'symbol':file.replace('.csv',''), 'strategy':'ML-CNC', 'return':ret, 'win_rate':win,
                              'avg_hold':hold, 'sharpe':sharpe, 'trades':len(trades)})

        # Williams %R signal backtest
        df_wr = simple_williams_r_signal(df)
        trades, ret, win, hold, sharpe = backtest_cnc(df_wr, 'signal', allow_early_exit=True)
        final_results.append({'symbol':file.replace('.csv',''), 'strategy':'Williams_R-CNC', 'return':ret,
                              'win_rate':win, 'avg_hold':hold, 'sharpe':sharpe, 'trades':len(trades)})

    # You can similarly add 1h and 15m timeframe analysis for ML and Williams R, adjusting data loading and signal generation accordingly

    # Summarize
    df_res = pd.DataFrame(final_results)
    print(df_res.groupby('strategy').agg({'return':'mean','win_rate':'mean','avg_hold':'mean','sharpe':'mean','trades':'sum'}).round(2))

if __name__ == "__main__":
    run_full_retest()
