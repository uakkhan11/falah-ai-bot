import os
import logging
import pandas as pd
import numpy as np
import talib
import ta
from xgboost import XGBClassifier

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# ---------- Config ----------
BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data")
}
YEAR_FILTER = 2025

# Strategy toggles (start simple, add later)
STRAT_TREND_BREAKOUT = True
STRAT_WITH_TREND_PULLBACK = False
STRAT_EVENT_CONTINUATION = False

# Universe curation (optional)
SYMBOL_WHITELIST = set([])  # e.g., {"INFIBEAM","HINDCOPPER","TDPOWERSYS"}
SYMBOL_BLACKLIST = set(["LLOYDSENGG","SAGILITY","WELSPUNLIV"])

# Entries and filters (loosened)
VOLUME_MULT_BREAKOUT = 1.5
ADX_THRESHOLD_BREAKOUT = 25
ADX_THRESHOLD_DEFAULT = 20
HOURLY_ADX_MIN = 22     # was 25
DAILY_ADX_MIN = 18      # was 20

# Exits
ATR_SL_MULT = 1.3
PROFIT_TARGET = 0.02
TRAIL_TRIGGER = 0.02
PARTIAL_TP_FRACTION = 0.5

# Risk & capacity
MAX_TRADES = 500
MAX_POSITIONS = 5
INITIAL_CAPITAL = 100000
MIN_TRADE_VALUE = 1.0
MAX_POSITION_FRACTION = 0.2
BASE_RISK = 0.02

# Holding horizon
MAX_HOLD_BARS = 48

# Costs & execution
ROUND_TRIP_BPS = 0.002
ADV_PARTICIPATION = 0.02
ENTRY_AT_NEXT_BAR = True

# ML gate (loosened composite)
FEATURES = ["adx","atr","volume_ratio","adosc","hour_adx","volume_sma","macd_hist","vwap","roc","obv"]
ML_PROBA_THRESHOLD = 0.5
ML_COMPOSITE_MIN = 0.55   # was 0.6

# Portfolio layer
DAILY_LOSS_LIMIT = -0.015
MAX_NEW_ENTRIES_PER_15M = 3

# ---------- Data loading ----------
def load_and_filter_2025(symbol):
    symbol = str(symbol).strip().strip("()[]'")
    def filter_year(df):
        df['date'] = pd.to_datetime(df['date'])
        return df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)
    paths = {
        'daily': os.path.join(DATA_PATHS['daily'], f"{symbol}.csv"),
        '1hour': os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv"),
        '15minute': os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"),
    }
    missing = [k for k,p in paths.items() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing data files for {symbol}: {missing}")
    daily = pd.read_csv(paths['daily'])
    hourly = pd.read_csv(paths['1hour'])
    m15 = pd.read_csv(paths['15minute'])
    return filter_year(daily), filter_year(hourly), filter_year(m15)

def add_indicators(df):
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])

    # Weekly Donchian
    df_weekly = df.set_index('date').resample('W-MON').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna().reset_index()
    df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20, min_periods=1).max()
    df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high'] \
        .reindex(df['date'], method='ffill').values

    # Trend & volatility
    df['donchian_high'] = df['high'].rolling(20, min_periods=1).max()
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    adx_df = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_df.adx()
    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()

    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()

    # RSI and VWAP
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum().replace(0, np.nan))

    # ATR & chandelier
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    atr_ce = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=22).average_true_range()
    high22 = df['high'].rolling(22, min_periods=1).max()
    df['chandelier_exit'] = high22 - 3.0 * atr_ce

    # TA-Lib series
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)

    macd, macd_signal, macd_hist = talib.MACD(close)
    df['macd_hist'] = macd_hist
    df['roc'] = talib.ROC(close, timeperiod=10)
    df['obv'] = talib.OBV(close, volume)
    df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

    for f in ["hour_adx", "adosc", "roc", "obv", "vwap", "volume_sma", "ema50"]:
        if f not in df.columns:
            df[f] = np.nan
    return df

def add_hourly_features_to_m15(m15_df, hourly_df):
    hourly_df = hourly_df.copy()
    m15_df = m15_df.copy()
    hourly_df['date'] = pd.to_datetime(hourly_df['date'])
    m15_df['date'] = pd.to_datetime(m15_df['date'])
    hourly_df = hourly_df.set_index('date')
    m15_df = m15_df.set_index('date')
    for col in ['adx','atr','macd_hist']:
        if col in hourly_df.columns:
            m15_df['hour_'+col] = hourly_df[col].reindex(m15_df.index, method='ffill')
    return m15_df.reset_index()

# ---------- Strategy signals (with datetime alignment) ----------
def trend_breakout_signal(df, daily_df, hourly_df):
    df = df.copy(); daily_df = daily_df.copy(); hourly_df = hourly_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    hourly_df['date'] = pd.to_datetime(hourly_df['date'])
    m15_idx = df['date']

    # Daily gate
    d_ok = ((daily_df['ema50'] > daily_df['ema200']) & (daily_df['adx'] >= DAILY_ADX_MIN)).rename('d_ok')
    d_ok.index = daily_df['date']
    daily_gate = d_ok.reindex(m15_idx, method='ffill').fillna(False)

    # Hourly gate
    h_ok_series = (hourly_df.set_index('date')['adx'] >= HOURLY_ADX_MIN)
    hourly_gate = h_ok_series.reindex(m15_idx, method='ffill').fillna(False)

    # 15m breakout with volume & weekly context
    cond_don = df['close'] > df['donchian_high'].shift(1)
    cond_vol = df['volume'] > VOLUME_MULT_BREAKOUT * df['vol_sma20']
    # cond_wk = df['close'] > df['weekly_donchian_high'].shift(1)
    # brk = cond_don & cond_vol & cond_wk
    brk = cond_don & cond_vol

    sig = (brk & daily_gate & hourly_gate).astype(int)
    return sig

def with_trend_pullback_signal(df, daily_df, hourly_df):
    # disabled by toggle; keep function for later use
    df = df.copy(); daily_df = daily_df.copy(); hourly_df = hourly_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    hourly_df['date'] = pd.to_datetime(hourly_df['date'])
    m15_idx = df['date']

    d_ok = ((daily_df['ema50'] > daily_df['ema200']) & (daily_df['adx'] >= DAILY_ADX_MIN)).rename('d_ok')
    d_ok.index = daily_df['date']
    daily_gate = d_ok.reindex(m15_idx, method='ffill').fillna(False)
    h_ok_series = (hourly_df.set_index('date')['adx'] >= HOURLY_ADX_MIN)
    hourly_gate = h_ok_series.reindex(m15_idx, method='ffill').fillna(False)

    ema20 = ta.trend.ema_indicator(df['close'], window=20)
    ema50_15 = ta.trend.ema_indicator(df['close'], window=50)
    rsi = df['rsi']
    pulled = (df['low'] <= ema20) | (df['low'] <= ema50_15)
    reclaim = (df['close'] > df['vwap']) & (df['close'] > ema20)
    rsi_ok = (rsi >= 35) & (rsi <= 50)
    obv_rising = df['obv'].diff() > 0
    adosc_pos = df['adosc'] > 0

    sig = (pulled.shift(1).fillna(False) & reclaim & rsi_ok & obv_rising & adosc_pos & daily_gate & hourly_gate).astype(int)
    return sig

def event_continuation_signal(df, daily_df, hourly_df):
    # disabled by toggle; keep function for later use
    df = df.copy(); daily_df = daily_df.copy(); hourly_df = hourly_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    hourly_df['date'] = pd.to_datetime(hourly_df['date'])
    m15_idx = df['date']

    d_ok_series = (daily_df['ema50'] > daily_df['ema200'])
    d_ok_series.index = daily_df['date']
    d_ok = d_ok_series.reindex(m15_idx, method='ffill').fillna(False)
    h_ok_series = (hourly_df.set_index('date')['adx'] >= HOURLY_ADX_MIN)
    h_ok = h_ok_series.reindex(m15_idx, method='ffill').fillna(False)

    x = df.copy()
    x['date_only'] = pd.to_datetime(x['date']).dt.date
    day_meds = x.groupby('date_only')['volume'].transform('median')
    first_hour = pd.to_datetime(x['date']).dt.hour == 9
    event_day = (first_hour & (x['volume'] > 2.5 * day_meds)).groupby(x['date_only']).transform('max')
    event_gate = pd.Series(event_day.values, index=x.index).reindex(x.index, method='ffill').fillna(False)

    day_high = x.groupby('date_only')['high'].transform('cummax')
    after_first_hour = pd.to_datetime(x['date']).dt.hour >= 10
    vol_ok = x['volume'] > VOLUME_MULT_BREAKOUT * x['vol_sma20']
    cont = after_first_hour & vol_ok & (x['close'] > day_high.shift(1))

    sig = (event_gate & cont & d_ok & h_ok).astype(int)
    return sig

def build_entry_signals(m15, daily, hourly):
    m15 = m15.copy()
    m15['entry_signal'] = 0
    m15['entry_type'] = ''

    if STRAT_TREND_BREAKOUT:
        tb = trend_breakout_signal(m15, daily, hourly)
        m15.loc[tb == 1, ['entry_signal','entry_type']] = [1, 'Trend_Breakout']

    if STRAT_WITH_TREND_PULLBACK:
        wtp = with_trend_pullback_signal(m15, daily, hourly)
        idx = (wtp == 1) & (m15['entry_signal'] == 0)
        m15.loc[idx, ['entry_signal','entry_type']] = [1, 'WithTrend_Pullback']

    if STRAT_EVENT_CONTINUATION:
        ec = event_continuation_signal(m15, daily, hourly)
        idx = (ec == 1) & (m15['entry_signal'] == 0)
        m15.loc[idx, ['entry_signal','entry_type']] = [1, 'Event_Continuation']

    return m15

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

        hour_adx = df.loc[val_idx, 'hour_adx'].fillna(0)
        align_factor = (hour_adx / 50.0).clip(upper=1.0)
        composite = proba.loc[val_idx].fillna(0) * align_factor
        gate.loc[val_idx] = ((proba.loc[val_idx] >= ML_PROBA_THRESHOLD) & (composite >= ML_COMPOSITE_MIN)).astype(int)

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

# ---------- Backtest with portfolio risk ----------
def backtest_live_like(df, symbol, capital=INITIAL_CAPITAL, day_loss_limit=DAILY_LOSS_LIMIT):
    cash = capital
    positions = {}
    trades = []
    trade_count = 0
    df = df.copy().reset_index(drop=True)

    # Portfolio-level trackers (per day)
    df['day'] = pd.to_datetime(df['date']).dt.date
    daily_realized_pnl = {}
    new_entries_window = {}

    def can_enter(now_dt):
        # Daily loss check (realized PnL only)
        day = now_dt.date()
        realized = daily_realized_pnl.get(day, 0.0)
        if realized < day_loss_limit * capital:
            return False
        # Entry throttle per 15m window
        window_key = (day, now_dt.hour, now_dt.minute//15)
        count = new_entries_window.get(window_key, 0)
        if count >= MAX_NEW_ENTRIES_PER_15M:
            return False
        return True

    for i in range(1, len(df)):
        row = df.iloc[i]
        date = pd.to_datetime(row['date'])
        price = float(row['close'])
        sig = int(row.get('entry_signal', 0))
        etype = row.get('entry_type', '')

        # Exits
        to_close = []
        for pid, pos in list(positions.items()):
            ret = (price - pos['entry_price']) / pos['entry_price']
            bars_held = i - pos['entry_idx']
            atr_stop = pos['entry_price'] - ATR_SL_MULT * pos['entry_atr']
            pos['high'] = max(pos['high'], price)
            # trail logic
            if (not pos['trail_active']) and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = row['chandelier_exit']
            if pos['trail_active'] and row['chandelier_exit'] > pos['trail_stop']:
                pos['trail_stop'] = row['chandelier_exit']

            # Partial TP
            if (not pos.get('partial_taken', False)) and ret >= PROFIT_TARGET:
                part_shares = pos['shares'] * PARTIAL_TP_FRACTION
                sell_val = part_shares * price
                buy_val = part_shares * pos['entry_price']
                costs = ROUND_TRIP_BPS * (buy_val + sell_val)
                pnl = sell_val - buy_val - costs
                cash += sell_val
                trades.append({
                    'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': date,
                    'pnl': pnl, 'entry_type': pos['entry_type'], 'exit_reason': 'Partial_TP'
                })
                pos['shares'] -= part_shares
                pos['partial_taken'] = True
                daily_realized_pnl[date.date()] = daily_realized_pnl.get(date.date(), 0.0) + pnl
                continue

            # Full exit
            exit_reason = None
            if price <= atr_stop:
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
                    'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': date,
                    'pnl': pnl, 'entry_type': pos['entry_type'], 'exit_reason': exit_reason
                })
                to_close.append(pid)
                trade_count += 1
                daily_realized_pnl[date.date()] = daily_realized_pnl.get(date.date(), 0.0) + pnl

        for pid in to_close:
            positions.pop(pid, None)
        if trade_count >= MAX_TRADES:
            break

        # Entries
        if sig == 1 and len(positions) < MAX_POSITIONS and cash > 0 and can_enter(date):
            prob = float(row.get('ml_proba', 0.5))
            atr = float(row['atr']) if not np.isnan(row['atr']) else 1.0
            size_val = dynamic_position_sizing(prob, atr, cash)
            # Liquidity cap
            bar_val = float(row['close']) * float(row['volume'])
            liq_cap = ADV_PARTICIPATION * bar_val
            size_val = min(size_val, liq_cap)
            if size_val < MIN_TRADE_VALUE:
                continue

            exec_price = price
            exec_idx = i
            if ENTRY_AT_NEXT_BAR and i+1 < len(df):
                exec_price = float(df.iloc[i+1].get('open', df.iloc[i+1]['close']))
                exec_idx = i+1

            shares = size_val / exec_price
            buy_val = shares * exec_price
            entry_costs = ROUND_TRIP_BPS * buy_val
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
                'entry_idx': exec_idx,
                'partial_taken': False
            }
            window_key = (date.date(), date.hour, date.minute//15)
            new_entries_window[window_key] = new_entries_window.get(window_key, 0) + 1

    return trades

# ---------- Orchestration ----------
def prepare_data(symbol):
    daily, hourly, m15 = load_and_filter_2025(symbol)
    daily = add_indicators(daily)
    hourly = add_indicators(hourly)
    m15 = add_indicators(m15)
    m15 = add_hourly_features_to_m15(m15, hourly)
    # build entries
    m15 = build_entry_signals(m15, daily, hourly)
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
    # m15_ml['entry_signal'] = ((m15_ml['entry_signal'] == 1) & (m15_ml['ml_signal'] == 1)).astype(int)
    m15_ml['entry_signal'] = (m15_ml['entry_signal'] == 1).astype(int)
    return m15_ml

# ---------- Summary Reporting ----------
def summarize_and_save(trades_df, stats_df):
    total_trades = len(trades_df)
    wins = int((trades_df['pnl'] > 0).sum())
    losses = int((trades_df['pnl'] <= 0).sum())
    win_rate = round((wins / total_trades) * 100, 2) if total_trades else 0.0
    total_pnl = round(trades_df['pnl'].sum(), 2) if total_trades else 0.0
    avg_pnl = round(trades_df['pnl'].mean(), 2) if total_trades else 0.0

    by_exit = (trades_df.groupby('exit_reason')['pnl']
               .agg(['count','sum','mean'])
               .rename(columns={'count':'trades','sum':'total_pnl','mean':'avg_pnl'})
               .reset_index())
    by_exit.to_csv('walk_forward_by_exit_v21.csv', index=False)

    by_entry = (trades_df.groupby('entry_type')['pnl']
               .agg(['count','sum','mean'])
               .rename(columns={'count':'trades','sum':'total_pnl','mean':'avg_pnl'})
               .reset_index())
    by_entry.to_csv('walk_forward_by_entry_v21.csv', index=False)

    stats_df.to_csv('walk_forward_stats_v21.csv', index=False)

    print('\n===== Walk-Forward Summary (v2.1 loosened) =====')
    print(f'Total trades: {total_trades}')
    print(f'Wins: {wins} | Losses: {losses} | Win rate: {win_rate}%')
    print(f'Total PnL: {total_pnl} | Avg PnL/trade: {avg_pnl}')
    print('Top 5 symbols by Total PnL:')
    if not stats_df.empty and 'Total PnL' in stats_df.columns:
        print(stats_df.sort_values('Total PnL', ascending=False).head(5))
    else:
        print('(no stats)')
    print('\nBy exit reason (saved to walk_forward_by_exit_v21.csv):')
    print(by_exit.head())
    print('\nBy entry type (saved to walk_forward_by_entry_v21.csv):')
    print(by_entry.head())

def run_walk_forward(symbols):
    # Apply universe curation
    if SYMBOL_WHITELIST:
        symbols = [s for s in symbols if s in SYMBOL_WHITELIST]
    if SYMBOL_BLACKLIST:
        symbols = [s for s in symbols if s not in SYMBOL_BLACKLIST]

    all_stats, all_trades = [], []
    logging.info(f"Discovered {len(symbols)} symbols (post-curation)")
    for idx, symbol in enumerate(symbols, 1):
        logging.info(f"[{idx}/{len(symbols)}] {symbol}")
        try:
            daily, hourly, m15 = prepare_data(symbol)
        except FileNotFoundError as e:
            logging.warning(str(e))
            continue
        if m15.empty:
            logging.warning(f"No 2025 data after warmup for {symbol}; skipping")
            continue
        m15_ml = walk_forward_predict_gate(m15, hourly)
        n_sig = int(m15_ml['entry_signal'].sum())
        logging.info(f"Signals: {n_sig}")
        trades = backtest_live_like(m15_ml, symbol)
        logging.info(f"Trades: {len(trades)}")
        stats = extract_trade_stats(trades)
        stats.update({'Symbol': symbol, 'Strategy': 'ML+Regime Walk-Forward v2.1 loosened'})
        all_stats.append(stats)
        for t in trades:
            t['Symbol'] = symbol
        all_trades.extend(trades)

    stats_df = pd.DataFrame(all_stats)
    trades_df = pd.DataFrame(all_trades)

    # Save primary CSVs
    stats_df.to_csv('walk_forward_stats_v21.csv', index=False)
    trades_df.to_csv('walk_forward_trades_v21.csv', index=False)

    # Print and save summary/breakdowns
    if not trades_df.empty:
        summarize_and_save(trades_df, stats_df)
    else:
        print("No trades generated; summary skipped.")

    logging.info(f"Completed. Symbols processed: {len(stats_df)}; Total trades: {len(trades_df)}")
    return stats_df, trades_df

if __name__ == '__main__':
    files = [f for f in os.listdir(DATA_PATHS['daily']) if f.lower().endswith('.csv')]
    symbols = []
    for f in files:
        base, _ = os.path.splitext(f)
        base = base.strip().strip("()[]'")
        if base:
            symbols.append(base)
    logging.info(f"Symbols discovered (pre-curation): {len(symbols)}. Example: {symbols[:5]}")
    stats_df, trades_df = run_walk_forward(symbols)
    logging.info("Saved walk_forward_stats_v21.csv, walk_forward_trades_v21.csv, and breakdown CSVs")
