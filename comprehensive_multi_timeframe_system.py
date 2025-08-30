# comprehensive_ml_all_timeframes.py

import os
import pandas as pd
import numpy as np
import ta
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour':    os.path.join(BASE_DIR, "intraday_swing_data"),
    'daily':    os.path.join(BASE_DIR, "swing_data"),
}

TIMEFRAME_CONFIGS = {
    '1minute':  {'source': '15minute', 'agg': 1, 'pt': 0.015, 'sl': 0.010, 'mh': 60},
    '5minute':  {'source': '15minute', 'agg': 3, 'pt': 0.020, 'sl': 0.012, 'mh': 36},
    '15minute': {'source': '15minute', 'agg': 1, 'pt': 0.025, 'sl': 0.015, 'mh': 25},
    '1hour':    {'source': '1hour',    'agg': 1, 'pt': 0.035, 'sl': 0.020, 'mh': 24},
    '4hour':    {'source': '1hour',    'agg': 4, 'pt': 0.050, 'sl': 0.025, 'mh': 12},
    'daily':    {'source': 'daily',    'agg': 1, 'pt': 0.060, 'sl': 0.030, 'mh': 10},
}

ML_MODEL = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

def list_symbols():
    return [f[:-4] for f in os.listdir(DATA_PATHS['daily']) if f.endswith('.csv')]

def load_resample(symbol, cfg):
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
            bars.append({
                'date': chunk['date'].iloc[-1],
                'open': chunk['open'].iloc[0],
                'high': chunk['high'].max(),
                'low': chunk['low'].min(),
                'close': chunk['close'].iloc[-1],
                'volume': chunk['volume'].sum()
            })
        df = pd.DataFrame(bars)
    return df if len(df) >= 50 else None

def make_features(df):
    df = df.copy().reset_index(drop=True)
    df['ret'] = df['close'].pct_change().fillna(0)
    df['rsi'] = ta.momentum.rsi(df['close'], 14).fillna(50)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14).fillna(25)
    df['sma20'] = df['close'].rolling(20).mean().fillna(method='bfill')
    df['vol_r'] = df['volume'] / df['volume'].rolling(20).mean().fillna(1)
    return df.fillna(method='bfill').fillna(method='ffill')

def make_target(df, cfg):
    tgt = []
    for i in range(len(df) - cfg['mh']):
        e = df['close'].iloc[i]
        win = False
        for j in range(1, cfg['mh'] + 1):
            f = df['close'].iloc[i + j]
            r = (f - e) / e
            if r >= cfg['pt']:
                tgt.append(1)
                win = True
                break
            if r <= -cfg['sl']:
                tgt.append(0)
                win = True
                break
        if not win:
            f = df['close'].iloc[min(i + cfg['mh'], len(df) - 1)]
            tgt.append(1 if (f - e) / e > 0 else 0)
    tgt.extend([0] * cfg['mh'])
    return np.array(tgt)

def train_backtest(symbols, timeframe):
    cfg = TIMEFRAME_CONFIGS[timeframe]
    Xs, ys = [], []
    for s in symbols:
        df = load_resample(s, cfg)
        if df is None:
            continue
        df = make_features(df)
        y = make_target(df, cfg)
        X = df[['ret', 'rsi', 'adx', 'sma20', 'vol_r']].values
        mask = ~np.isnan(X).any(axis=1)
        Xs.append(X[mask])
        ys.append(y[mask])
    if not Xs:
        print(f"{timeframe}: No data")
        return
    Xall = np.vstack(Xs)
    yall = np.concatenate(ys)
    if len(np.unique(yall)) < 2:
        print(f"{timeframe}: Only one class present")
        return
    scaler = StandardScaler().fit(Xall)
    Xs = scaler.transform(Xall)
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, yall, test_size=0.3, random_state=42, stratify=yall)
    ML_MODEL.fit(Xtr, ytr)
    test_acc = ML_MODEL.score(Xte, yte)
    preds = ML_MODEL.predict(Xs)
    trades = pd.DataFrame({'pred': preds, 't': yall})
    wins = (trades[trades['pred'] == 1]['t'] == 1).sum()
    tot = len(trades[trades['pred'] == 1])
    win_pct = (wins / tot * 100) if tot else 0.0
    print(f"{timeframe}: Acc {test_acc*100:.1f}%, Trades {tot}, Wins {wins} ({win_pct:.1f}%)")

if __name__ == "__main__":
    symbols = list_symbols()
    for tf in ['1minute', '5minute', '15minute', '1hour', '4hour', 'daily']:
        print(f"=== ML {tf} ===")
        train_backtest(symbols, tf)
