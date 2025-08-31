# parallel_ensemble_backtest.py

import os
import pandas as pd
import numpy as np
import ta
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

# Configuration
BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour':    os.path.join(BASE_DIR, "intraday_swing_data"),
    'daily':    os.path.join(BASE_DIR, "swing_data"),
}
TIMEFRAME_CONFIGS = {
    '1minute':  {'source': '15minute', 'agg': 1, 'pt': 0.015, 'sl': 0.010, 'mh': 60, 'weight': 0.5},
    '5minute':  {'source': '15minute', 'agg': 3, 'pt': 0.020, 'sl': 0.012, 'mh': 36, 'weight': 0.7},
    '15minute': {'source': '15minute', 'agg': 1, 'pt': 0.025, 'sl': 0.015, 'mh': 25, 'weight': 1.0},
    '1hour':    {'source': '1hour',    'agg': 1, 'pt': 0.035, 'sl': 0.020, 'mh': 24, 'weight': 1.3},
    '4hour':    {'source': '1hour',    'agg': 4, 'pt': 0.050, 'sl': 0.025, 'mh': 12, 'weight': 1.5},
    'daily':    {'source': 'daily',    'agg': 1, 'pt': 0.060, 'sl': 0.030, 'mh': 10, 'weight': 1.7},
}
ENSEMBLE_THRESHOLD = 4.0
POSITION_RISK = 0.02  # 2% of capital per trade

# Utilities
def list_symbols():
    return [f[:-4] for f in os.listdir(DATA_PATHS['daily']) if f.endswith('.csv')]

def load_resample(symbol, tf):
    cfg = TIMEFRAME_CONFIGS[tf]
    path = os.path.join(DATA_PATHS[cfg['source']], f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year == 2025].reset_index(drop=True)
    if len(df) < 50:
        return None
    if cfg['agg'] > 1:
        bars = []
        for i in range(0, len(df) - cfg['agg'] + 1, cfg['agg']):
            chunk = df.iloc[i:i+cfg['agg']]
            if len(chunk) == cfg['agg']:
                bars.append({
                    'date':   chunk['date'].iloc[-1],
                    'open':   chunk['open'].iloc[0],
                    'high':   chunk['high'].max(),
                    'low':    chunk['low'].min(),
                    'close':  chunk['close'].iloc[-1],
                    'volume': chunk['volume'].sum()
                })
        df = pd.DataFrame(bars)
    return df if len(df) >= 50 else None

def create_features(df):
    df = df.copy()
    df['ret']   = df['close'].pct_change().fillna(0)
    df['rsi']   = ta.momentum.rsi(df['close'], window=14).fillna(50)
    df['adx']   = ta.trend.adx(df['high'], df['low'], df['close'], window=14).fillna(25)
    df['sma20'] = df['close'].rolling(20).mean().fillna(method='bfill')
    df['vol_r'] = df['volume'] / df['volume'].rolling(20).mean().fillna(1)
    return df.fillna(method='bfill').fillna(method='ffill')

def train_model_for_tf(symbol, tf):
    df = load_resample(symbol, tf)
    if df is None:
        return None, None
    df = create_features(df)
    cfg = TIMEFRAME_CONFIGS[tf]
    tgt = np.zeros(len(df))
    for i in range(len(df) - cfg['mh']):
        entry = df['close'].iloc[i]
        window = df['close'].iloc[i+1:i+1+cfg['mh']]
        if (window >= entry*(1+cfg['pt'])).any():
            tgt[i] = 1
        elif (window <= entry*(1-cfg['sl'])).any():
            tgt[i] = 0
        else:
            tgt[i] = int(window.iloc[-1] > entry)
    X = df[['ret','rsi','adx','sma20','vol_r']].values
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], tgt[mask]
    if len(np.unique(y))<2 or len(X)<100:
        return None, None
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    return model, scaler

# Prepare models and scalers once (using first symbol for demo; ideally aggregate)
models, scalers = {}, {}
first_symbol = list_symbols()[0]
for tf in TIMEFRAME_CONFIGS:
    m, s = train_model_for_tf(first_symbol, tf)
    models[tf], scalers[tf] = m, s

def backtest_symbol(symbol):
    trades = []
    # Load and feature each timeframe
    data = {}
    for tf in TIMEFRAME_CONFIGS:
        df = load_resample(symbol, tf)
        if df is not None:
            data[tf] = create_features(df)
    if '1minute' not in data:
        return trades
    N = len(data['1minute'])
    for i in range(20, N):
        score = 0
        for tf, cfg in TIMEFRAME_CONFIGS.items():
            df = data.get(tf)
            if df is None or i>=len(df):
                continue
            raw = df.iloc[i][['ret','rsi','adx','sma20','vol_r']].values
            try:
                feat = np.array(raw, dtype=float)
            except:
                continue
            if np.isnan(feat).any():
                continue
            pred = models[tf].predict(scalers[tf].transform([feat]))[0]
            score += cfg['weight'] * pred
        if score >= ENSEMBLE_THRESHOLD:
            price = data['1minute'].iloc[i]['close']
            trades.append({'symbol': symbol, 'entry_idx': i, 'entry_price': price, 'score': score})
    return trades

if __name__ == "__main__":
    symbols = list_symbols()
    with Pool(processes=cpu_count()) as pool:
        result_lists = pool.map(backtest_symbol, symbols)
    all_trades = [trade for sub in result_lists for trade in sub]
    df = pd.DataFrame(all_trades)
    df.to_csv("ensemble_trades.csv", index=False)
    print(f"Completed: {len(all_trades)} trades across {len(symbols)} symbols")
