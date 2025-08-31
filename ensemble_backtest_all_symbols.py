# ensemble_backtest_all_symbols.py

import os
import pandas as pd
import numpy as np
import ta
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# --------------------------
# CONFIGURATION
# --------------------------

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour':    os.path.join(BASE_DIR, "intraday_swing_data"),
    'daily':    os.path.join(BASE_DIR, "swing_data"),
}

TIMEFRAME_CONFIGS = {
    '1minute':  {'source': '15minute', 'agg': 1, 'pt': 0.015, 'sl': 0.010,  'mh': 60, 'weight': 0.5},
    '5minute':  {'source': '15minute', 'agg': 3, 'pt': 0.020, 'sl': 0.012,  'mh': 36, 'weight': 0.7},
    '15minute': {'source': '15minute', 'agg': 1, 'pt': 0.025, 'sl': 0.015,  'mh': 25, 'weight': 1.0},
    '1hour':    {'source': '1hour',    'agg': 1, 'pt': 0.035, 'sl': 0.020,  'mh': 24, 'weight': 1.3},
    '4hour':    {'source': '1hour',    'agg': 4, 'pt': 0.050, 'sl': 0.025,  'mh': 12, 'weight': 1.5},
    'daily':    {'source': 'daily',    'agg': 1, 'pt': 0.060, 'sl': 0.030,  'mh': 10, 'weight': 1.7},
}

ENSEMBLE_THRESHOLD = 4.0
INITIAL_CAPITAL = 100000
POSITION_RISK = 0.02  # 2% of capital per trade

# --------------------------
# UTILITY FUNCTIONS
# --------------------------

def list_symbols():
    path = DATA_PATHS['daily']
    return [f[:-4] for f in os.listdir(path) if f.endswith('.csv')]

def load_and_resample(symbol, tf):
    cfg = TIMEFRAME_CONFIGS[tf]
    file_path = os.path.join(DATA_PATHS[cfg['source']], f"{symbol}.csv")
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
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

def create_target(df, cfg):
    tgt = np.zeros(len(df))
    for i in range(len(df) - cfg['mh']):
        entry = df['close'].iloc[i]
        window = df['close'].iloc[i+1:i+1+cfg['mh']]
        hit_pt = (window >= entry*(1+cfg['pt'])).any()
        hit_sl = (window <= entry*(1-cfg['sl'])).any()
        if hit_pt:
            tgt[i] = 1
        elif hit_sl:
            tgt[i] = 0
        else:
            tgt[i] = int(window.iloc[-1] > entry)
    return tgt

def train_model(symbol, tf):
    df = load_and_resample(symbol, tf)
    if df is None: return None
    df = create_features(df)
    target = create_target(df, TIMEFRAME_CONFIGS[tf])
    X = df[['ret','rsi','adx','sma20','vol_r']].values
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], target[mask]
    if len(np.unique(y))<2 or len(X)<100: return None
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3,
                                           random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    return model, scaler

# --------------------------
# MAIN ENSEMBLE BACKTEST
# --------------------------

if __name__ == "__main__":
    symbols = list_symbols()
    # Train per-timeframe models
    models = {}
    scalers = {}
    for tf in TIMEFRAME_CONFIGS:
        print(f"Training model for {tf}")
        # Train on combined data from all symbols
        # Here, we simply train separately per timeframe using first symbol
        # For production, consider aggregated training
        m, s = train_model(symbols[0], tf)
        models[tf], scalers[tf] = m, s

    capital = INITIAL_CAPITAL
    trades = []

    # Iterate by symbol
    for symbol in symbols:
        # Load all timeframe data for this symbol
        symbol_data = {}
        for tf in TIMEFRAME_CONFIGS:
            df = load_and_resample(symbol, tf)
            if df is not None:
                symbol_data[tf] = create_features(df)

        # Align by longest index length
        base_tf = '1minute'
        if base_tf not in symbol_data: continue
        N = len(symbol_data[base_tf])

        # Generate ensemble signals
        for i in range(20, N):
            score = 0
            for tf, cfg in TIMEFRAME_CONFIGS.items():
                df = symbol_data.get(tf)
                if df is None or i>=len(df): continue
                feat = df.iloc[i][['ret','rsi','adx','sma20','vol_r']].values
                if np.isnan(feat).any(): continue
                feat_s = scalers[tf].transform([feat])
                pred = models[tf].predict(feat_s)[0]
                score += cfg['weight'] * pred

            # Entry condition
            if score >= ENSEMBLE_THRESHOLD and capital>0:
                entry_price = symbol_data[base_tf].iloc[i]['close']
                position_size = capital * POSITION_RISK
                shares = position_size / entry_price
                trades.append({
                    'symbol': symbol,
                    'timeframe': 'ensemble',
                    'entry_idx': i,
                    'entry_price': entry_price,
                    'shares': shares,
                    'score': score
                })
                capital -= position_size

        # Exit all positions at end for simplicity
    # Summarize
    print(f"Ensemble backtest completed. Trades generated: {len(trades)}")
    # Save trades
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv("ensemble_trades.csv", index=False)
    print("Saved ensemble_trades.csv")
