#!/usr/bin/env python3
import os, glob, math, random
import numpy as np
import pandas as pd
import talib as ta
import datetime
from dataclasses import dataclass

# 1) Configuration
BASE_DIR = "/root/falah-ai-bot"

DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '5minute': os.path.join(BASE_DIR, "five_minute_data"),
}

RESULTS_DIR = os.path.join(BASE_DIR, "results_live_like")
os.makedirs(RESULTS_DIR, exist_ok=True)

def merge_5m_features_onto_15m(df_15m, df_5m):
    # Compute 5-min features
    df_5m['rsi14_5m'] = ta.RSI(df_5m['close'], length=14)  # Use your RSI function
    df_5m['ema5_5m'] = ta.EMA(df_5m['close'], length=5)
    df_5m['ema20_5m'] = ta.EMA(df_5m['close'], length=20)

    # Example volume surge flag: 5m vol > 5-period SMA vol
    df_5m['vol_sma5'] = df_5m['volume'].rolling(window=5).mean()
    df_5m['vol_surge_5m'] = df_5m['volume'] > df_5m['vol_sma5']

    # Resample or map 5m features to 15m timestamp index (e.g. take last 5m bar within 15m bar)
    # Here assume df_15m and df_5m have datetime indices aligned
    features_5m = df_5m[['rsi14_5m', 'ema5_5m', 'ema20_5m', 'vol_surge_5m']].resample('15T').last()

    # Join features
    df_15m = df_15m.join(features_5m, how='left')

    return df_15m

# 2) Data Loading
def read_ohlcv_csv(path):
    assert isinstance(path, str), f"Expected string path, got {type(path)}: {path}"
    df = pd.read_csv(path)

    # Map columns case-insensitively
    cmap = {c.lower(): c for c in df.columns}

    # Select timestamp column
    ts_col = cmap.get('datetime') or cmap.get('date')
    if not ts_col:
        # fallback: use first column name as string
        ts_col = df.columns[0]

    # Parse timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')

    # Standard OHLCV columns case-insensitive
    open_src = cmap.get('open')
    high_src = cmap.get('high')
    low_src = cmap.get('low')
    close_src = cmap.get('close')
    vol_src = cmap.get('volume')

    missing = [name for name, col in [('open', open_src), ('high', high_src), ('low', low_src),
                                       ('close', close_src), ('volume', vol_src)] if col is None]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    df = df.rename(columns={
        ts_col: 'datetime',
        open_src: 'open',
        high_src: 'high',
        low_src: 'low',
        close_src: 'close',
        vol_src: 'volume'
    })

    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].dropna()
    df = df.sort_values('datetime').reset_index(drop=True)
    return df.set_index('datetime')

def discover_symbols():
    dir15 = DATA_PATHS['15minute']
    files_15 = glob.glob(os.path.join(dir15, "*.csv"))

    print(f"[discover_symbols] 15m dir: {dir15}")
    print(f"[discover_symbols] count: {len(files_15)}")
    if files_15[:5]:
        print("[discover_symbols] sample files:", [os.path.basename(p) for p in files_15[:5]])

    if not files_15:
        print("[discover_symbols] No 15m files found.")
        return []

    df_5m = load_5m_data(...)
    df_15m = load_15m_data(...)
    
    # Merge 5m features onto 15m
    df_15m = merge_5m_features_onto_15m(df_15m, df_5m)
    
    # Now start backtesting with df_15m (which now includes 5m features)
    results = backtest_strategy(df_15m)

    # Build roots
    roots = [os.path.basename(p).rsplit(".",1)[0] for p in files_15]
    # For filenames like RELIANCE_15m.csv, take before first underscore
    symbols = [r.split('_')[0] for r in roots]
    # Deduplicate preserving order
    symbols = list(dict.fromkeys(symbols))

    # Sanity checks
    assert all(isinstance(s, str) for s in symbols), f"Non-string symbol in discover: {symbols[:5]}"
    assert all(s for s in symbols), f"Empty symbol in discover: {symbols[:5]}"

    print(f"[discover_symbols] Found {len(symbols)} symbols, sample: {symbols[:10]}")
    return symbols

def load_frames(symbol):
    def pick_one(folder):
        hits = glob.glob(os.path.join(folder, f"{symbol}*.csv"))
        if not hits:
            print(f"[load_frames v2] {symbol}: no match in {folder}")
            return None
        if len(hits) > 1:
            print(f"[load_frames v2] {symbol}: multiple matches in {folder}, picking {os.path.basename(hits[0])}")
        return hits[0]  # only the first match path

    print(f"[load_frames v2] selecting paths for {symbol}")
    p15 = pick_one(DATA_PATHS['15minute'])
    p1h = pick_one(DATA_PATHS['1hour'])
    pdly = pick_one(DATA_PATHS['daily'])

    if not (p15 and p1h and pdly):
        print(f"[load_frames v2] Missing files for {symbol}: 15m={p15} 1h={p1h} 1d={pdly}")
        return None

    return read_ohlcv_csv(p15), read_ohlcv_csv(p1h), read_ohlcv_csv(pdly)

# 3) Indicators & Features
def ema(x, span): return x.ewm(span=span, adjust=False).mean()

def rsi(x, period=14):
    d = x.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    rg = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rl = dn.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = rg / rl.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def macd(x, fast=12, slow=26, signal=9):
    m = ema(x, fast) - ema(x, slow)
    s = ema(m, signal)
    h = m - s
    return m, s, h

def bollinger(x, window=20, k=2.0):
    m = x.rolling(window).mean()
    sd = x.rolling(window).std(ddof=0)
    return m, m + k * sd, m - k * sd

def atr(df, period=14):
    pc = df['close'].shift(1)
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - pc).abs(),
        (df['low'] - pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def vwap_session(df, session='D'):
    g = df.groupby(pd.Grouper(freq=session))
    pv = g.apply(lambda x: (x['close'] * x['volume']).cumsum()).reset_index(level=0, drop=True)
    vv = g['volume'].cumsum()
    return pv / vv

def resample_no_lookahead(lower_df, higher_df, cols):
    return higher_df[cols].shift(1).reindex(lower_df.index, method='ffill')

def opening_range(df, minutes=15):
    bars = max(1, minutes // 15)
    grp = df.groupby(pd.Grouper(freq='D'))
    or_hi = grp['high'].transform(lambda x: x.iloc[:bars].max() if len(x) else np.nan)
    or_lo = grp['low'].transform(lambda x: x.iloc[:bars].min() if len(x) else np.nan)
    return or_hi, or_lo

def build_features(df15, df1h, dfd):
    X = df15.copy()
    X['vwap'] = vwap_session(X, 'D')
    X['ema9'] = ema(X['close'], 9)
    X['ema21'] = ema(X['close'], 21)
    X['bb_mid'], X['bb_up'], X['bb_dn'] = bollinger(X['close'], 20, 2.0)
    X['rsi14'] = rsi(X['close'], 14)
    X['macd'], X['macd_sig'], X['macd_hist'] = macd(X['close'])
    X['atr14'] = atr(X, 14)
    X['or_hi'], X['or_lo'] = opening_range(X, 15)

    H = df1h.copy()
    H['ema9_h'] = ema(H['close'], 9)
    H['ema21_h'] = ema(H['close'], 21)
    H['rsi14_h'] = rsi(H['close'], 14)

    D = dfd.copy()
    D['ema200_d'] = ema(D['close'], 200)
    D['rsi14_d'] = rsi(D['close'], 14)

    X = X.join(resample_no_lookahead(X, H, ['ema9_h','ema21_h','rsi14_h']))
    X = X.join(resample_no_lookahead(X, D, ['ema200_d','rsi14_d']))
    return X

# 4) Signal gates
def signal_gates_row(r):
    gates = {}
    gate_ltf_entry = False
    if 'rsi14_5m' in r and 'ema5_5m' in r and 'ema20_5m' in r and 'vol_surge_5m' in r:
        gate_ltf_entry = (
            (r['ema5_5m'] > r['ema20_5m']) and (r['rsi14_5m'] > 55)
        ) or (r['vol_surge_5m'] == True)
    else:
        # If 5m data missing, default to True to not block entries
        gate_ltf_entry = True

    # Update long_signal to include lower timeframe gate
    long_signal = long_signal and gate_ltf_entry
    gates['regime_up'] = (r['close'] > r['ema200_d']) and (r['rsi14_d'] >= 55)  # strong uptrends[1]
    gates['tf1_up'] = (r['ema9_h'] > r['ema21_h'])
    gates['tf1_dn'] = (r['ema9_h'] < r['ema21_h'])
    gates['value_long'] = (r['close'] >= r['vwap']) and (r['close'] >= r['ema21'])
    gates['value_short'] = (r['close'] <= r['vwap']) and (r['close'] <= r['ema21'])
    gates['pullback_long'] = True
    gates['pullback_short'] = (r['close'] >= r['ema9']*0.99)
    gates['momo_long'] = (r['macd'] > r['macd_sig']) and (r['rsi14'] > 52)
    gates['momo_short'] = (r['macd'] < r['macd_sig']) and (r['rsi14'] < 48)
    gates['orb_long'] = (r['close'] > r['or_hi']) if not pd.isna(r['or_hi']) else False
    gates['orb_short'] = (r['close'] < r['or_lo']) if not pd.isna(r['or_lo']) else False
    or_width_ok = False
    if pd.notna(r.get('or_hi')) and pd.notna(r.get('or_lo')):
        or_width_ok = ((r['or_hi'] - r['or_lo']) / max(1e-9, r['close'])) >= 0.001
    long_signal = (
        gates['regime_up']
        and gates['tf1_up']
        and gates['value_long']
        and gates['orb_long']
        and gates['momo_long']
        and or_width_ok
    )

    # Lower timeframe confirmation gate (5-minute)
    gate_ltf_entry = False
    if 'rsi14_5m' in r and 'ema5_5m' in r and 'ema20_5m' in r and 'vol_surge_5m' in r:
        gate_ltf_entry = (
            (r['ema5_5m'] > r['ema20_5m']) and (r['rsi14_5m'] > 55)
        ) or (r['vol_surge_5m'] == True)
    else:
        gate_ltf_entry = True

    # Combine with previous
    long_signal = long_signal and gate_ltf_entry

    short_signal = False

    return bool(long_signal), bool(short_signal), gates

# 5) Event classes and execution model
@dataclass
class MarketEvent:
    time: pd.Timestamp
    o: float; h: float; l: float; c: float; v: float

# Configuration constants (place these at the very top)
RISK_PER_TRADE = 0.01

ATR_MULT_STOP = 1.5

TARGET_R = 1.5

COMMISSION_BPS = 1.0

SPREAD_BP = 1.0

SLIP_MEAN_BP = 2.0

SLIP_STD_BP = 1.0

class ExecModel:
    def __init__(self, commission_bps=COMMISSION_BPS, spread_bp=SPREAD_BP,
                 slip_mean_bp=SLIP_MEAN_BP, slip_std_bp=SLIP_STD_BP):
        self.com_bps = commission_bps
        self.spread_bp = spread_bp
        self.slip_mu = slip_mean_bp
        self.slip_sd = slip_std_bp
    def fee(self, notional): return notional * (self.com_bps/10000.0)
    def apply_spread(self, ref, side):
        mult = 1 + (self.spread_bp/10000.0) * (1 if side=='buy' else -1)
        return ref * mult
    def slippage_bps(self): return max(0.0, random.gauss(self.slip_mu, self.slip_sd))

# 6) Live-like backtester
class LiveLikeBacktester:
    def __init__(self, symbol, df15, df1h, dfd, cash=1_000_000):
        self.symbol = symbol
        self.cash = cash
        self.equity = cash
        self.exec = ExecModel()
        self.feats = build_features(df15, df1h, dfd)
        self.df = df15.loc[self.feats.index]
        self.pos = 0
        self.qty = 0
        self.avg = np.nan
        self.stop = np.nan
        self.target = np.nan
        self.ledger = []
        self.equity_curve = []
        self._last_gates = {}

    def size_from_atr(self, open_px, atr_val):
        if np.isnan(atr_val) or atr_val <= 0:
            return 0, 0.0
        stop_dist = max(atr_val * ATR_MULT_STOP, open_px*0.002)
        risk_cash = self.equity * RISK_PER_TRADE
        qty = int(risk_cash // stop_dist)
        return max(qty, 0), stop_dist

    def on_bar(self, t, row):
        feats = self.feats.loc[t]
        r = {**feats.to_dict(), 'open': row['open'], 'high': row['high'],
             'low': row['low'], 'close': row['close']}
        long_sig, short_sig, gates = signal_gates_row(r)
        bar_dt = pd.Timestamp(t)
        if getattr(bar_dt, "tzinfo", None) is not None:
            bar_dt = bar_dt.tz_convert(None).tz_localize(None)
        bar_time = bar_dt.to_pydatetime().time()
        
        # Extended session times: early 09:15-11:15 and late 14:30-15:30
        allow_opening = (datetime.time(9, 15) <= bar_time <= datetime.time(11, 15))
        allow_power = (datetime.time(14, 30) <= bar_time <= datetime.time(15, 30))
        
        if not (allow_opening or allow_power):
            long_sig = False
            short_sig = False
        self._last_gates = gates
        # exits
        if self.pos != 0:
            if self.pos > 0:
                if row['low'] <= self.stop:
                    px = self.exec.apply_spread(self.stop, 'sell')
                    slip = self.exec.slippage_bps(); px *= 1 - slip/10000.0
                    fee = self.exec.fee(px * self.qty)
                    pnl = self.qty * (px - self.avg) - fee
                    self.equity += pnl
                    self.ledger.append({'time': t,'symbol': self.symbol,'side': 'sell',
                        'reason': 'stop','qty': self.qty,'price': px,'fee': fee,'slip_bp': slip,'pnl': pnl,'equity': self.equity})
                    self.pos, self.qty, self.avg = 0, 0, np.nan
                elif row['high'] >= self.target:
                    px = self.exec.apply_spread(self.target, 'sell')
                    slip = self.exec.slippage_bps(); px *= 1 - slip/10000.0
                    fee = self.exec.fee(px * self.qty)
                    pnl = self.qty * (px - self.avg) - fee
                    self.equity += pnl
                    self.ledger.append({'time': t,'symbol': self.symbol,'side': 'sell',
                        'reason': 'target','qty': self.qty,'price': px,'fee': fee,'slip_bp': slip,'pnl': pnl,'equity': self.equity})
                    self.pos, self.qty, self.avg = 0, 0, np.nan
            else:
                if row['high'] >= self.stop:
                    px = self.exec.apply_spread(self.stop, 'buy')
                    slip = self.exec.slippage_bps(); px *= 1 + slip/10000.0
                    fee = self.exec.fee(px * self.qty)
                    pnl = (self.avg - px) * self.qty - fee
                    self.equity += pnl
                    self.ledger.append({'time': t,'symbol': self.symbol,'side': 'buy',
                        'reason': 'stop','qty': self.qty,'price': px,'fee': fee,'slip_bp': slip,'pnl': pnl,'equity': self.equity})
                    self.pos, self.qty, self.avg = 0, 0, np.nan
                elif row['low'] <= self.target:
                    px = self.exec.apply_spread(self.target, 'buy')
                    slip = self.exec.slippage_bps(); px *= 1 + slip/10000.0
                    fee = self.exec.fee(px * self.qty)
                    pnl = (self.avg - px) * self.qty - fee
                    self.equity += pnl
                    self.ledger.append({'time': t,'symbol': self.symbol,'side': 'buy',
                        'reason': 'target','qty': self.qty,'price': px,'fee': fee,'slip_bp': slip,'pnl': pnl,'equity': self.equity})
                    self.pos, self.qty, self.avg = 0, 0, np.nan
        return long_sig, short_sig

    def run(self):
        pending_entry = None
        for i, (t, row) in enumerate(self.df.iterrows()):
            if pending_entry is not None and self.pos == 0:
                side = pending_entry
                atrv = self.feats.loc[t, 'atr14']
                open_px = row['open']
                qty, stop_dist = self.size_from_atr(open_px, atrv)
                if qty > 0:
                    gates_pref = {f"gate_{k}": bool(v) for k,v in self._last_gates.items()}
                    if side == 'long':
                        px = self.exec.apply_spread(open_px, 'buy')
                        slip = self.exec.slippage_bps(); px *= 1 + slip/10000.0
                        fee = self.exec.fee(px * qty); self.equity -= fee
                        self.pos, self.qty, self.avg = 1, qty, px
                        self.stop = px - stop_dist
                        self.target = px + TARGET_R*stop_dist
                        self.ledger.append({'time': t,'symbol': self.symbol,'side': 'buy',
                            'reason': 'signal', 'qty': qty,'price': px,'fee': fee,'slip_bp': slip,'pnl': 0.0,'equity': self.equity, **gates_pref})
                    else:
                        px = self.exec.apply_spread(open_px, 'sell')
                        slip = self.exec.slippage_bps(); px *= 1 - slip/10000.0
                        fee = self.exec.fee(px * qty); self.equity -= fee
                        self.pos, self.qty, self.avg = -1, qty, px
                        self.stop = px + stop_dist
                        self.target = px - TARGET_R*stop_dist
                        self.ledger.append({'time': t,'symbol': self.symbol,'side': 'sell',
                            'reason': 'signal', 'qty': qty,'price': px,'fee': fee,'slip_bp': slip,'pnl': 0.0,'equity': self.equity, **gates_pref})
                pending_entry = None
            long_sig, short_sig = self.on_bar(t, row)
            if self.pos == 0:
                if long_sig: pending_entry = 'long'
                elif short_sig: pending_entry = 'short'
                else: pending_entry = None

            # mark-to-market
            mtm = 0.0
            if self.pos != 0:
                if self.pos > 0: mtm = self.qty * (row['close'] - self.avg)
                else: mtm = self.qty * (self.avg - row['close'])
            self.equity_curve.append({'time': t,'symbol': self.symbol,'equity': self.equity + mtm,
                'cash': self.equity, 'pos': self.pos,'qty': self.qty})

        # close at the end
        if self.pos != 0:
            t = self.df.index[-1]
            px = self.df['close'].iloc[-1]
            fee = self.exec.fee(px * self.qty)
            if self.pos > 0:
                pnl = self.qty * (px - self.avg) - fee
                self.ledger.append({'time': t,'symbol': self.symbol,'side': 'sell','reason':'eod',
                    'qty': self.qty,'price': px,'fee': fee,'slip_bp': 0.0,'pnl': pnl,'equity': self.equity + pnl})
            else:
                pnl = (self.avg - px) * self.qty - fee
                self.ledger.append({'time': t,'symbol': self.symbol,'side': 'buy','reason':'eod',
                    'qty': self.qty,'price': px,'fee': fee,'slip_bp': 0.0,'pnl': pnl,'equity': self.equity + pnl})
            self.equity += pnl
            self.pos = 0
            self.qty = 0

        return pd.DataFrame(self.ledger), pd.DataFrame(self.equity_curve)

# 7) Reports
def build_reports(trades_path, out_dir):
    df = pd.read_csv(trades_path, parse_dates=['time'])
    if df.empty:
        for name in ["indicator_report.csv","combo_report.csv","ablation_report.csv"]:
            pd.DataFrame([]).to_csv(os.path.join(out_dir, name), index=False)
        return

    df = df.sort_values(['symbol','time']).reset_index(drop=True)
    df['is_entry'] = (df['reason'] == 'signal').astype(int)
    df['trade_id'] = df.groupby('symbol')['is_entry'].cumsum()
    df.loc[df['trade_id'] == 0, 'trade_id'] = np.nan

    gate_cols = [c for c in df.columns if c.startswith('gate_')]

    entries = df[df['reason'] == 'signal'][['symbol','trade_id'] + gate_cols].copy()
    exits = df[df['reason'].isin(['stop','target','eod'])].copy()

    if exits.empty:
        for name in ["indicator_report.csv","combo_report.csv","ablation_report.csv"]:
            pd.DataFrame([]).to_csv(os.path.join(out_dir, name), index=False)
        return

    trades = exits.merge(entries, on=['symbol','trade_id'], how='left', suffixes=('_exit','_entry'))

    gate_cols = [c for c in entries.columns if c.startswith('gate_')]
    
    for g in gate_cols:
        
        if f"{g}_entry" in trades:
            
            trades[g] = trades[f"{g}_entry"] # take entry-time gate value
    
    drop_cols = [c for c in trades.columns if c.endswith('_entry') or c.endswith('_exit')]
    
    trades = trades.drop(columns=drop_cols)



    trades['label_profit'] = (trades['pnl'] > 0).astype(int)
    trades['R_proxy'] = trades['pnl'] / (trades['price'].abs() * trades['qty'].replace(0,np.nan))

    def profit_factor(series):
        pos = series[series > 0].sum()
        neg = -series[series < 0].sum()
        return pos/neg if neg > 0 else np.nan

    rows = []
    base_hit = trades['label_profit'].mean() if len(trades) else np.nan
    for g in gate_cols:
        sub = trades[trades[g] == True]
        if len(sub) == 0:
            continue
        rows.append({
            'indicator': g,
            'n_trades': len(sub),
            'win_rate': sub['label_profit'].mean(),
            'profit_factor': profit_factor(sub['pnl']),
            'avg_R_proxy': sub['R_proxy'].mean(),
            'precision': sub['label_profit'].mean(),
            'recall': (sub['label_profit'].sum() / max(1, trades['label_profit'].sum())) if len(trades) else np.nan,
            'lift_over_baseline': (sub['label_profit'].mean()/base_hit) if base_hit not in [0, np.nan] else np.nan
        })
    ind_rep = pd.DataFrame(rows).sort_values(['profit_factor','win_rate'], ascending=False) if rows else pd.DataFrame([])
    ind_rep.to_csv(os.path.join(out_dir, "indicator_report.csv"), index=False)

    from itertools import combinations
    rows = []
    for a, b in combinations(gate_cols, 2):
        sub = trades[(trades[a] == True) & (trades[b] == True)]
        if len(sub) < 20:
            continue
        rows.append({'combo': f"{a}+{b}",
            'n_trades': len(sub),
            'win_rate': sub['label_profit'].mean(),
            'profit_factor': profit_factor(sub['pnl']),
            'avg_R_proxy': sub['R_proxy'].mean(),
            'lift_over_baseline': (sub['label_profit'].mean()/base_hit) if base_hit not in [0, np.nan] else np.nan
        })
    combo_rep = pd.DataFrame(rows).sort_values(['profit_factor','win_rate'], ascending=False) if rows else pd.DataFrame([])
    combo_rep.to_csv(os.path.join(out_dir, "combo_report.csv"), index=False)

    def metrics(d):
        return pd.Series({
            'n_trades': len(d),
            'win_rate': d['label_profit'].mean() if len(d) else np.nan,
            'avg_R_proxy': d['R_proxy'].mean() if len(d) else np.nan,
            'profit_factor': profit_factor(d['pnl']) if len(d) else np.nan,
        })

    base_m = metrics(trades)
    rows = []
    for g in gate_cols:
        sub = trades[~(trades[g] == True)]
        m = metrics(sub)
        rows.append({'drop_gate': g,
            'base_win_rate': base_m['win_rate'],
            'base_profit_factor': base_m['profit_factor'],
            'base_avg_R_proxy': base_m['avg_R_proxy'],
            'drop_win_rate': m['win_rate'],
            'drop_profit_factor': m['profit_factor'],
            'drop_avg_R_proxy': m['avg_R_proxy'],
            'delta_win': (m['win_rate'] - base_m['win_rate']) if pd.notna(base_m['win_rate']) else np.nan,
            'delta_profit_factor': (m['profit_factor'] - base_m['profit_factor']) if pd.notna(base_m['profit_factor']) else np.nan,
            'delta_avg_R_proxy': (m['avg_R_proxy'] - base_m['avg_R_proxy']) if pd.notna(base_m['avg_R_proxy']) else np.nan,
        })
    abl_rep = pd.DataFrame(rows).sort_values(['delta_profit_factor','delta_win'], ascending=False) if rows else pd.DataFrame([])
    abl_rep.to_csv(os.path.join(out_dir, "ablation_report.csv"), index=False)

# 8) Run the whole universe and save results
def run_universe(symbols, cash=1_000_000):
    all_trades, all_equity = [], []
    print(f"Symbols discovered: {len(symbols)} -> {symbols[:8]}")  # diagnostic info
    for sym in symbols:
        frames = load_frames(sym)
        if frames is None:
            print(f"[skip] {sym}: missing one of 15m/1h/daily files")
            continue
        df15, df1h, dfd = frames

        start = max(df15.index.min(), df1h.index.min(), dfd.index.min())
        end   = min(df15.index.max(), df1h.index.max(), dfd.index.max())

        df15 = df15[(df15.index >= start) & (df15.index <= end)]
        df1h = df1h[(df1h.index >= start) & (df1h.index <= end)]
        dfd  = dfd[(dfd.index >= start) & (dfd.index <= end)]

        print(f"[info] {sym}: 15m={len(df15)} 1h={len(df1h)} dly={len(dfd)} window=({start} -> {end})")

        if len(df15) < 300 or start >= end:
            print(f"[skip] {sym}: insufficient overlap or bars")
            continue

        # Instantiate backtest
        eng = LiveLikeBacktester(sym, df15, df1h, dfd, cash=cash)
        trades, eq = eng.run()
        if not trades.empty:
            all_trades.append(trades)
        if not eq.empty:
            all_equity.append(eq)

    # Save results
    trades_df = pd.concat(all_trades).reset_index(drop=True) if all_trades else pd.DataFrame()
    eq_df = pd.concat(all_equity).reset_index(drop=True) if all_equity else pd.DataFrame()
    return trades_df, eq_df

if __name__ == "__main__":
    syms = discover_symbols()
    trades, eq = run_universe(syms)
    trades.to_csv(os.path.join(RESULTS_DIR, "trades.csv"), index=False)
    eq.to_csv(os.path.join(RESULTS_DIR, "equity_curve.csv"), index=False)
    eq[['time','symbol','pos','qty']].to_csv(os.path.join(RESULTS_DIR, "positions.csv"), index=False)
    build_reports(os.path.join(RESULTS_DIR, "trades.csv"), RESULTS_DIR)
    print("Wrote trades.csv, equity_curve.csv, positions.csv, indicator_report.csv, combo_report.csv, ablation_report.csv to", RESULTS_DIR)
