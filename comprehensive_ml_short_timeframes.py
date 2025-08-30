# comprehensive_ml_short_timeframes.py

import os
import pandas as pd
import numpy as np
import ta
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
warnings.filterwarnings("ignore")

# =============================================================================
# ML SYSTEM FOR SHORT TIMEFRAMES (1m, 5m, 15m)
# =============================================================================

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}

TIMEFRAME_CONFIGS = {
    '1minute': {'source': '15minute', 'agg_bars': 1, 'profit_target': 0.015, 'stop_loss': 0.010, 'max_hold': 60},
    '5minute': {'source': '15minute', 'agg_bars': 3, 'profit_target': 0.020, 'stop_loss': 0.012, 'max_hold': 36},
    '15minute': {'source': '15minute', 'agg_bars': 1, 'profit_target': 0.025, 'stop_loss': 0.015, 'max_hold': 25},
}

ML_MODEL = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

def list_symbols():
    path = DATA_PATHS['15minute']
    return [f[:-4] for f in os.listdir(path) if f.endswith('.csv')]

def load_resample(symbol, cfg):
    df = pd.read_csv(os.path.join(DATA_PATHS[cfg['source']], f"{symbol}.csv"))
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year == 2025].reset_index(drop=True)
    if cfg['agg_bars'] > 1:
        bars = []
        for i in range(0, len(df) - cfg['agg_bars'] + 1, cfg['agg_bars']):
            chunk = df.iloc[i:i+cfg['agg_bars']]
            bars.append({
                'date': chunk['date'].iloc[-1],
                'open': chunk['open'].iloc[0],
                'high': chunk['high'].max(),
                'low': chunk['low'].min(),
                'close': chunk['close'].iloc[-1],
                'volume': chunk['volume'].sum()
            })
        df = pd.DataFrame(bars)
    return df

def make_features(df):
    df = df.copy()
    df['return'] = df['close'].pct_change().fillna(0)
    df['rsi'] = ta.momentum.rsi(df['close'], 14).fillna(50)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14).fillna(25)
    df['sma20'] = df['close'].rolling(20).mean().fillna(method='bfill')
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean().fillna(1)
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

def make_target(df, cfg):
    tgt = np.zeros(len(df))
    for i in range(len(df) - cfg['max_hold']):
        start = df['close'].iloc[i]
        window = df['close'].iloc[i+1:i+1+cfg['max_hold']]
        if (window >= start*(1+cfg['profit_target'])).any():
            tgt[i] = 1
        elif (window <= start*(1-cfg['stop_loss'])).any():
            tgt[i] = 0
        else:
            tgt[i] = int(window.iloc[-1] > start)
    return tgt

def train_and_backtest(symbols, timeframe):
    cfg = TIMEFRAME_CONFIGS[timeframe]
    X_list, y_list = [], []
    for s in symbols:
        df = load_resample(s, cfg)
        if len(df) < 50: continue
        df = make_features(df)
        y = make_target(df, cfg)
        X = df[['return','rsi','adx','sma20','vol_ratio']].values
        mask = ~np.isnan(X).any(axis=1)
        X_list.append(X[mask])
        y_list.append(y[mask])
    if not X_list: return
    Xall = np.vstack(X_list); yall = np.concatenate(y_list)
    if len(np.unique(yall))<2: return
    scaler = StandardScaler().fit(Xall)
    Xall = scaler.transform(Xall)
    Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.3, random_state=42, stratify=yall)
    ML_MODEL.fit(Xtr, ytr)
    test_score = ML_MODEL.score(Xte, yte)
    print(f"{timeframe}: Test Accuracy {test_score*100:.1f}%")
    # Simple backtest: simulate trades where model predicts 1
    preds = ML_MODEL.predict(Xall)
    df_all = pd.DataFrame({'pred':preds,'target':yall})
    trades = df_all[df_all['pred']==1]
    wins = (trades['target']==1).sum(); total = len(trades)
    print(f" - Trades: {total}, Wins: {wins} ({(wins/total*100) if total else 0:.1f}%)")

if __name__ == "__main__":
    symbols = list_symbols()[:50]
    for tf in ['1minute','5minute','15minute']:
        print(f"=== Testing ML {tf} ===")
        train_and_backtest(symbols, tf)
