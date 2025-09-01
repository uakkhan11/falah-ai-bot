import os
import pandas as pd
import numpy as np
import talib
import ta
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
from xgboost import XGBClassifier

# ---------- Config ----------
BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data")
}
YEAR_FILTER = 2025
VOLUME_MULT_BREAKOUT = 2
ADX_THRESHOLD_BREAKOUT = 25
ADX_THRESHOLD_DEFAULT = 20
ATR_SL_MULT = 1.0
PROFIT_TARGET = 0.03
TRAIL_TRIGGER = 0.01
MAX_TRADES = 500
MAX_POSITIONS = 5
INITIAL_CAPITAL = 100000
MIN_TRADE_VALUE = 1.0
FEATURES = ["adx","atr","volume_ratio","adosc","hour_adx","volume_sma","macd_hist","vwap","roc","obv"]
ML_PROBA_THRESHOLD = 0.4
MAX_POSITION_FRACTION = 0.2
BASE_RISK = 0.02
MAX_HOLD_BARS = 20
ROUND_TRIP_BPS = 0.002   # 20 bps round trip (slippage + fees)
ADV_PARTICIPATION = 0.02 # max 2% of bar volume value
ENTRY_AT_NEXT_BAR = True

# ---------- Helpers ----------
def load_and_filter_2025(symbol):
    symbol = str(symbol).strip().strip("()[]'")  # ensure clean string

    def filter_year(df):
        df['date'] = pd.to_datetime(df['date'])
        return df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)

    paths = {
        'daily': os.path.join(DATA_PATHS['daily'], f"{symbol}.csv"),
        '1hour': os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv"),
        '15minute': os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"),
    }

    # Option A: skip symbols missing any timeframe
    missing = [k for k, p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing data files for {symbol}: {missing}")

    daily = pd.read_csv(paths['daily'])
    hourly = pd.read_csv(paths['1hour'])
    m15 = pd.read_csv(paths['15minute'])

    return filter_year(daily), filter_year(hourly), filter_year(m15)


def add_indicators(df):
    df = df.sort_values('date').reset_index(drop=True)
    df_weekly = df.set_index('date').resample('W-MON').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
    df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20, min_periods=1).max()
    df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high'].reindex(df['date'], method='ffill').values
    df['donchian_high'] = df['high'].rolling(20, min_periods=1).max()
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    adx_df = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_df.adx()
    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband(); df['bb_lower'] = bb.bollinger_lband()
    high14 = df['high'].rolling(14).max(); low14 = df['low'].rolling(14).min()
    df['wpr'] = (high14 - df['close']) / (high14 - low14) * -100
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    atr_ce = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=22).average_true_range()
    high20 = df['high'].rolling(22, min_periods=1).max()
    df['chandelier_exit'] = high20 - 3.0 * atr_ce
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    macd, macd_signal, macd_hist = talib.MACD(close)
    df['macd_hist'] = macd_hist
    df['roc'] = talib.ROC(close, timeperiod=10)
    df['obv'] = talib.OBV(close, volume)
    df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['volume_ratio'] = df['volume'] / df['vol_sma20']
    df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    return df
    for f in ["hour_adx", "adosc", "roc", "obv", "vwap"]:
    if f not in df.columns:
        df[f] = np.nan

def add_hourly_features_to_m15(m15_df, hourly_df):
    hourly_df = hourly_df.set_index('date')
    m15_df = m15_df.set_index('date')
    for col in ['adx','atr','macd_hist']:
        if col in hourly_df.columns:
            m15_df['hour_'+col] = hourly_df[col].reindex(m15_df.index, method='ffill')
    return m15_df.reset_index()

def breakout_signal(df):
    cond_d = df['close'] > df['donchian_high'].shift(1)
    cond_v = df['volume'] > VOLUME_MULT_BREAKOUT * df['vol_sma20']
    cond_w = df['close'] > df['weekly_donchian_high'].shift(1)
    df['breakout_signal'] = (cond_d & cond_v & cond_w).astype(int)
    return df

def bb_breakout_signal(df):
    df['bb_breakout_signal'] = ((df['close'] > df['bb_upper']) & (df['volume'] > VOLUME_MULT_BREAKOUT * df['vol_sma20'])).astype(int)
    return df

def bb_pullback_signal(df):
    cond_pull = df['close'] < df['bb_lower']
    cond_resume = df['close'] > df['bb_lower'].shift(1)
    df['bb_pullback_signal'] = (cond_pull.shift(1) & cond_resume).astype(int)
    return df

def combine_signals(df):
    chand_ok = (df['close'] > df['chandelier_exit'])
    regime_breakout = (df['close'] > df['ema200']) & (df['adx'] > ADX_THRESHOLD_BREAKOUT)
    regime_default = (df['close'] > df['ema200']) & (df['adx'] > ADX_THRESHOLD_DEFAULT)
    df['entry_signal'] = 0
    df['entry_type'] = ''
    df.loc[(df['breakout_signal'] == 1) & chand_ok & regime_breakout, ['entry_signal','entry_type']] = [1,'Breakout']
    df.loc[(df['bb_breakout_signal'] == 1) & chand_ok & regime_breakout & (df['entry_signal']==0), ['entry_signal','entry_type']] = [1,'BB_Breakout']
    df.loc[(df['bb_pullback_signal'] == 1) & chand_ok & regime_default & (df['entry_signal']==0), ['entry_signal','entry_type']] = [1,'BB_Pullback']
    return df

# ---------- ML ----------
def fit_xgb(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42,
                          max_depth=5, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, n_estimators=200)
    model.fit(X_train, y_train)
    return model

def walk_forward_predict(m15_df, hourly_df, period='M'):
    df = m15_df.copy()
    df['future_return'] = df['close'].shift(-10) / df['close'] - 1
    df['label'] = (df['future_return'] > 0.01).astype(int)
    df = df.dropna(subset=FEATURES + ['label']).copy()

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    months = sorted(df.index.to_period(period).unique())

    proba = pd.Series(index=df.index, dtype=float)
    gate = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(months)):
        train_end = months[i-1].end_time
        val_period = months[i]
        val_start, val_end = val_period.start_time, val_period.end_time
        train_idx = df.index <= train_end
        val_idx = (df.index >= val_start) & (df.index <= val_end)
        if train_idx.sum() < 500 or val_idx.sum() == 0:
            continue
        X_train = df.loc[train_idx, FEATURES]
        y_train = df.loc[train_idx, 'label']
        model = fit_xgb(X_train, y_train)
        proba.loc[val_idx] = model.predict_proba(df.loc[val_idx, FEATURES])[:, 1]
        gate.loc[val_idx] = (proba.loc[val_idx] >= ML_PROBA_THRESHOLD).astype(int)

    df['ml_proba'] = proba
    df['ml_signal'] = gate.fillna(0).astype(int)
    df.reset_index(inplace=True)
    return df

# ---------- Sizing ----------
def dynamic_position_sizing(prob, atr, capital):
    atr = float(atr) if atr and atr > 0 else 1.0
    size_val = (BASE_RISK * capital) / atr
    size_val *= max(float(prob), 0.5)
    size_val = min(size_val, capital * MAX_POSITION_FRACTION)
    return max(size_val, 0.0)

# ---------- Backtest ----------
def backtest_live_like(df, symbol, capital=INITIAL_CAPITAL):
    cash = capital
    positions = {}
    trades = []
    trade_count = 0
    df = df.copy().reset_index(drop=True)

    for i in range(1, len(df)):
        row = df.iloc[i]
        date = row['date']; price = float(row['close'])
        sig = int(row.get('entry_signal', 0)); etype = row.get('entry_type', '')

        # Exits first
        to_close = []
        for pid, pos in list(positions.items()):
            ret = (price - pos['entry_price']) / pos['entry_price']
            bars_held = i - pos['entry_idx']
            atr_stop = pos['entry_price'] - ATR_SL_MULT * pos['entry_atr']
            pos['high'] = max(pos['high'], price)
            if (not pos['trail_active']) and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = row['chandelier_exit']
            if pos['trail_active'] and row['chandelier_exit'] > pos['trail_stop']:
                pos['trail_stop'] = row['chandelier_exit']

            exit_reason = None
            if ret >= PROFIT_TARGET:
                exit_reason = 'Profit Target'
            elif price <= atr_stop:
                exit_reason = 'ATR Stop Loss'
            elif pos['trail_active'] and price <= pos['trail_stop']:
                exit_reason = 'Chandelier Exit'
            elif bars_held >= MAX_HOLD_BARS:
                exit_reason = 'Time Exit'

            if exit_reason:
                buy_val = pos['shares'] * pos['entry_price']
                sell_val = pos['shares'] * price
                costs = ROUND_TRIP_BPS * (buy_val + sell_val)
                pnl = sell_val - buy_val - costs
                cash += sell_val
                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'entry_type': pos['entry_type'],
                    'exit_reason': exit_reason
                })
                to_close.append(pid)
                trade_count += 1

        for pid in to_close:
            positions.pop(pid, None)
        if trade_count >= MAX_TRADES:
            break

        # Entries (next-bar execution)
        if sig == 1 and len(positions) < MAX_POSITIONS and cash > 0:
            prob = float(row.get('ml_proba', 0.5))
            atr = float(row['atr']) if not np.isnan(row['atr']) else 1.0
            size_val = dynamic_position_sizing(prob, atr, cash)

            # Liquidity cap: value traded <= ADV_PARTICIPATION * (close * volume)
            bar_val = float(row['close']) * float(row['volume'])
            liq_cap = ADV_PARTICIPATION * bar_val
            size_val = min(size_val, liq_cap)

            if size_val < MIN_TRADE_VALUE:
                continue

            exec_price = price
            exec_idx = i
            if ENTRY_AT_NEXT_BAR and i+1 < len(df):
                # Next bar open if available, else next close
                exec_price = float(df.iloc[i+1].get('open', df.iloc[i+1]['close']))
                exec_idx = i+1

            shares = size_val / exec_price
            buy_val = shares * exec_price
            entry_costs = ROUND_TRIP_BPS * buy_val  # entry side component included here
            if buy_val + entry_costs > cash:
                continue

            cash -= buy_val
            positions[len(positions)+1] = {
                'entry_date': df.iloc[exec_idx]['date'],
                'entry_price': exec_price,
                'shares': shares,
                'high': exec_price,
                'trail_active': False,
                'trail_stop': 0.0,
                'entry_atr': atr,
                'entry_type': etype,
                'entry_idx': exec_idx
            }

    return trades

# ---------- Orchestration ----------
def prepare_data(symbol):
    daily, hourly, m15 = load_and_filter_2025(symbol)
    daily = add_indicators(daily)
    hourly = add_indicators(hourly)
    m15 = add_indicators(m15)
    m15 = add_hourly_features_to_m15(m15, hourly)
    m15 = breakout_signal(m15)
    m15 = bb_breakout_signal(m15)
    m15 = bb_pullback_signal(m15)
    m15 = combine_signals(m15)
    # Warmup cut: drop until EMA200 available to avoid early-bar bias
    m15 = m15.dropna(subset=['ema200']).reset_index(drop=True)
    return daily, hourly, m15

def extract_trade_stats(trades):
    df = pd.DataFrame(trades)
    if df.empty:
        return {}
    return {
        'Total Trades': len(df),
        'Winning Trades': int((df['pnl'] > 0).sum()),
        'Losing Trades': int((df['pnl'] <= 0).sum()),
        'Win Rate %': round((df['pnl'] > 0).mean() * 100, 2),
        'Avg PnL per Trade': round(df['pnl'].mean(), 2),
        'Best PnL': round(df['pnl'].max(), 2),
        'Worst PnL': round(df['pnl'].min(), 2),
        'Total PnL': round(df['pnl'].sum(), 2)
    }

def walk_forward_predict_gate(m15, hourly):
    m15_ml = walk_forward_predict(m15, hourly)
    # Gate: require both strategy signal and ML gate
    m15_ml['entry_signal'] = ((m15_ml['entry_signal'] == 1) & (m15_ml['ml_signal'] == 1)).astype(int)
    return m15_ml

def run_walk_forward(symbols):
    all_stats, all_trades = [], []
    logging.info(f"Discovered {len(symbols)} symbols")
    for idx, symbol in enumerate(symbols, 1):
        logging.info(f"[{idx}/{len(symbols)}] Preparing {symbol}")
        try:
            _, hourly, m15 = prepare_data(symbol)
        except FileNotFoundError as e:
            logging.warning(str(e))
            continue
        if m15.empty:
            logging.warning(f"No 2025 data after warmup for {symbol}; skipping")
            continue
        logging.info(f"Walk-forward ML for {symbol} with {len(m15)} bars")
        m15_ml = walk_forward_predict_gate(m15, hourly)
        n_sig = int(m15_ml['entry_signal'].sum())
        logging.info(f"Signals for {symbol}: {n_sig}")
        trades = backtest_live_like(m15_ml, symbol)
        logging.info(f"Trades for {symbol}: {len(trades)}")
        stats = extract_trade_stats(trades)
        stats.update({'Symbol': symbol, 'Strategy': 'ML+All Features Walk-Forward'})
        all_stats.append(stats)
        for t in trades:
            t['Symbol'] = symbol
        all_trades.extend(trades)
    stats_df = pd.DataFrame(all_stats)
    trades_df = pd.DataFrame(all_trades)
    logging.info(f"Completed. Symbols processed: {len(stats_df)}; Total trades: {len(trades_df)}")
    stats_df.to_csv('walk_forward_stats.csv', index=False)
    trades_df.to_csv('walk_forward_trades.csv', index=False)
    return stats_df, trades_df


if __name__ == '__main__':
    files = [f for f in os.listdir(DATA_PATHS['daily']) if f.lower().endswith('.csv')]
    symbols = []
    for f in files:
        base, _ = os.path.splitext(f)
        base = base.strip().strip("()[]'")
        if base:
            symbols.append(base)
    logging.info(f"Symbols: {len(symbols)} discovered. Example: {symbols[:5]}")
    stats_df, trades_df = run_walk_forward(symbols)
    logging.info("Saved walk_forward_stats.csv and walk_forward_trades.csv")
