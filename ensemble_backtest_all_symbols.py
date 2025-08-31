# progress_ensemble_backtest_updated.py

import os
import pandas as pd
import numpy as np
import ta
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

warnings.filterwarnings("ignore")

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

ENSEMBLE_THRESHOLD = 3.0       # lowered from 4.0
CONFIDENCE_THRESHOLD = 0.55   # lowered from 0.65

def list_symbols():
    return [f[:-4] for f in os.listdir(DATA_PATHS['daily']) if f.endswith('.csv')]

def load_data(symbol, source):
    path = os.path.join(DATA_PATHS[source], f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df[df['date'].dt.year == 2025].reset_index(drop=True)

def simple_features(df):
    df = df.copy()
    df['ret']   = df['close'].pct_change().fillna(0)
    df['rsi']   = ta.momentum.rsi(df['close'], window=14).fillna(50)
    df['adx']   = ta.trend.adx(df['high'], df['low'], df['close'], window=14).fillna(25)
    df['sma20'] = df['close'].rolling(20).mean().fillna(method='bfill')
    df['vol_r'] = df['volume'] / df['volume'].rolling(20).mean().fillna(1)
    return df.fillna(method='bfill').fillna(0)

def print_progress(current, total, start_time, symbol=""):
    elapsed = time.time() - start_time
    if current > 0:
        eta = (elapsed / current) * (total - current)
        print(f"\r[{current/total*100:5.1f}%] {current}/{total} | "
              f"Elapsed: {elapsed/60:4.1f}m | ETA: {eta/60:4.1f}m | "
              f"Current: {symbol}", end="", flush=True)
    else:
        print(f"\r[  0.0%]   0/{total} | Starting...", end="", flush=True)

def train_global_model():
    print("\n🤖 Training global ML model...")
    X_all, y_all = [], []
    symbols = list_symbols()[:20]
    for i, sym in enumerate(symbols, 1):
        df = load_data(sym, '1hour')
        if df is None or len(df)<30:
            continue
        df = simple_features(df)
        y = (df['close'].shift(-1) > df['close']).astype(int)[:-1]
        X = df[['ret','rsi','adx','sma20','vol_r']].iloc[:-1].values
        mask = ~np.isnan(X).any(axis=1)
        X_all.append(X[mask])
        y_all.append(y[mask])
        print(f"  {i}/{len(symbols)} training symbols: {sym}")
    Xc = np.vstack(X_all); yc = np.concatenate(y_all)
    scaler = StandardScaler().fit(Xc)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(scaler.transform(Xc), yc)
    print("✅ Model trained on", len(yc), "samples\n")
    return model, scaler

def backtest_symbol(symbol, model, scaler):
    df = load_data(symbol, '1hour')
    if df is None:
        return []
    df = simple_features(df)
    trades = []
    for i in range(20, len(df)-1):
        raw = df.iloc[i][['ret','rsi','adx','sma20','vol_r']].values
        feat = np.array(raw, dtype=float)
        if np.isnan(feat).any():
            continue
        sc = scaler.transform([feat])
        pred = model.predict(sc)[0]
        conf = model.predict_proba(sc)[0][1]
        if pred==1 and conf>CONFIDENCE_THRESHOLD:
            # ensemble uses only the 1h model here; in full version you'd sum weights
            trades.append({
                'symbol': symbol,
                'entry_idx': i,
                'entry_price': df.iloc[i]['close'],
                'confidence': conf
            })
    return trades

if __name__=="__main__":
    start = time.time()
    model, scaler = train_global_model()
    syms = list_symbols()
    total = len(syms)
    all_trades=[]
    print(f"📊 Backtesting {total} symbols...")
    for idx, sym in enumerate(syms,1):
        print_progress(idx, total, start, sym)
        all_trades += backtest_symbol(sym, model, scaler)
        if idx%50==0:
            pd.DataFrame(all_trades).to_csv(f"temp_{idx}.csv", index=False)
    print_progress(total, total, start)
    print("\n💾 Saving final results...")
    pd.DataFrame(all_trades).to_csv("ensemble_trades_updated.csv", index=False)
    print(f"✅ Completed in {(time.time()-start)/60:.1f}m, trades: {len(all_trades)}")
