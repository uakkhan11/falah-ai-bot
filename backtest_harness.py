#!/usr/bin/env python3
import os
import csv
import json
import math
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# -------------------------
# Paths & config
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Output files
RULES_TRADES_CSV = os.path.join(OUT_DIR, "results_rules_15m.csv")
ML_TRADES_CSV = os.path.join(OUT_DIR, "results_ml_1h.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "summary_compare.csv")

# Universe and params
RULES_TIMEFRAME = "15m"
ML_TIMEFRAME = "1h"

# Rules strategy parameters (example)
RULES_CONF = {
    "profit_target": 0.02,   # 2%
    "stop_loss": 0.01,       # 1%
    "max_hold_bars": 24*4,   # e.g., 4 days worth of 15m bars (~96 bars)
}

# ML configuration (must match training artifacts/features)
ML_CONF = {
    "model_path": os.path.join(MODELS_DIR, "best_rf_1h.pkl"),
    "scaler_path": os.path.join(MODELS_DIR, "best_scaler_1h.pkl"),
    "feature_columns": ["ret", "rsi", "sma"],
    "proba_threshold": 0.65,
    "profit_target": 0.03,   # can be tuned
    "stop_loss": 0.015,
    "max_hold_bars": 48,     # e.g., 2 days on 1h bars
}

# -------------------------
# Data loading
# -------------------------
def list_symbols():
    # Use a stable set known to have data; extend as needed
    return sorted(list(set([
        "KANSAINER","BERGEPAINT","RHIM","WABAG","CONCORDBIO",
        "AXISCADES-BE","LUMAXTECH","POWERINDIA","SHARDACROP","GALLANTT",
        "KIOCL","SYRMA","TARIL","TRANSRAILL","WEBELSOLAR","FORCEMOT",
        "POKARNA","THYROCARE","YATHARTH","SKIPPER","PRIVISCL","SHARDAMOTR",
        "GABRIEL","PTCIL","NETWEB","LLOYDSENGG","BANCOINDIA","SHANTIGEAR",
        "TDPOWERSYS","CEWATER","WAAREEENER","SUNFLAG","WOCKPHARMA",
        "EMAMILTD","AVANTIFEED","LUXIND","FINPIPE","RAINBOW",
        "BHARTIARTL","HINDALCO","TORNTPHARM","ULTRACEMCO","OIL","JUBLFOOD","VOLTAS"
    ])))

def _synthetic_ohlcv(n=500, start="2025-01-01", freq="h", seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")  # use 'h' to avoid FutureWarning [1]
    ret = rng.normal(0, 0.002, size=n)
    price = 100 * np.exp(np.cumsum(ret))
    high = price * (1 + rng.uniform(0, 0.002, size=n))
    low = price * (1 - rng.uniform(0, 0.002, size=n))
    open_ = price * (1 + rng.uniform(-0.001, 0.001, size=n))
    close = price
    vol = rng.randint(1000, 5000, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)

def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    pq_path = os.path.join(DATA_DIR, timeframe, f"{symbol}.parquet")
    csv_path = os.path.join(DATA_DIR, timeframe, f"{symbol}.csv")
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp")  # [2]
    else:
        # Fallback synthetic so harness can run; seed per symbol + timeframe
        seed = (abs(hash(symbol + timeframe)) % 10_000)
        df = _synthetic_ohlcv(n=1000, freq="h" if timeframe == "1h" else "15min", seed=seed)

    df.columns = [c.lower() for c in df.columns]
    needed = ["open","high","low","close","volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol} {timeframe}: missing {missing}")
    df = df[needed].copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf,-np.inf], np.nan).dropna()
    return df

# -------------------------
# Feature engineering (align with trainer)
# -------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["close"].pct_change().fillna(0.0)
    delta = out["close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll = 14
    avg_gain = pd.Series(up, index=out.index).rolling(roll).mean().fillna(0.0)
    avg_loss = pd.Series(down, index=out.index).rolling(roll).mean().fillna(1e-9)
    rs = avg_gain / avg_loss
    out["rsi"] = 100 - (100 / (1 + rs))
    out["rsi"] = out["rsi"].fillna(50.0)
    out["sma"] = out["close"].rolling(20).mean().bfill()
    return out

# -------------------------
# Execution helpers
# -------------------------
def exit_trade(entry_price, highs, lows, profit_target, stop_loss, max_hold_bars):
    """
    Simulate an entry at t, then iterate forward to find exit by TP, SL, or time.
    highs/lows are forward arrays starting at the first bar after entry.
    Returns (exit_price, bars_held, exit_reason).
    """
    tp = entry_price * (1 + profit_target)
    sl = entry_price * (1 - stop_loss)
    bars = 0
    for h, l in zip(highs, lows):
        bars += 1
        if l <= sl:
            return sl, bars, "SL"
        if h >= tp:
            return tp, bars, "TP"
        if bars >= max_hold_bars:
            return highs.index[bars-1], bars, "TIME"  # overwrite below with close
    # Fallback on last bar close (shouldn’t hit if max_hold works)
    return highs.index[-1], bars, "TIME"

def summarize_trades(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"total_trades":0,"total_pnl":0.0,"win_rate":0.0,"profit_factor":0.0,"avg_pnl_per_trade":0.0,
                "best_trade":0.0,"worst_trade":0.0}
    total_trades = len(trades_df)
    total_pnl = trades_df["pnl"].sum()
    wins = (trades_df["pnl"] > 0).sum()
    losses = (trades_df["pnl"] < 0).sum()
    win_rate = 100.0 * wins / total_trades if total_trades else 0.0
    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
    avg_pnl = total_pnl / total_trades if total_trades else 0.0
    best_trade = trades_df["pnl"].max() if total_trades else 0.0
    worst_trade = trades_df["pnl"].min() if total_trades else 0.0
    return {
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl_per_trade": avg_pnl,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
    }

# -------------------------
# Rules 15m backtest
# -------------------------
def backtest_rules_15m(symbols):
    rows = []
    for sym in symbols:
        try:
            df = load_ohlcv(sym, RULES_TIMEFRAME)
        except Exception as e:
            print(f"[rules warn] {sym}: {e}")
            continue

        f = compute_features(df)
        fast = f["close"].rolling(10).mean()
        slow = f["close"].rolling(30).mean()
        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))

        # entries as 1-D integer positions
        entries = np.where(cross_up.to_numpy(dtype=bool))[0]  # take [0] to get the index array [web:257]

        for i in entries.tolist():  # iterate over plain Python ints
            if i + 1 >= len(df):
                continue

            entry_time = df.index[i]
            entry_price = float(df["close"].iloc[i])
            highs = df["high"].iloc[i+1:]
            lows = df["low"].iloc[i+1:]

            tp_price = entry_price * (1 + RULES_CONF["profit_target"])
            sl_price = entry_price * (1 - RULES_CONF["stop_loss"])

            exit_price = entry_price
            exit_reason = "TIME"
            bars_held = 0

            for j, (h, l) in enumerate(zip(highs.to_numpy(), lows.to_numpy()), start=1):
                if l <= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                    bars_held = j
                    break
                if h >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    bars_held = j
                    break
                if j >= RULES_CONF["max_hold_bars"]:
                    end_loc = min(i + j, len(df) - 1)
                    exit_price = float(df["close"].iloc[end_loc])
                    exit_reason = "TIME"
                    bars_held = j
                    break

            pnl = exit_price - entry_price
            rows.append([sym, entry_time, entry_price, exit_price, pnl, bars_held, exit_reason])

    cols = ["symbol","entry_time","entry_price","exit_price","pnl","bars_held","exit_reason"]
    trades = pd.DataFrame(rows, columns=cols)
    trades.to_csv(RULES_TRADES_CSV, index=False)
    return trades

# -------------------------
# ML 1h backtest
# -------------------------
def load_ml_artifacts():
    if not (os.path.exists(ML_CONF["model_path"]) and os.path.exists(ML_CONF["scaler_path"])):
        print("No ML trades or ML not loaded")
        return None, None
    model = joblib.load(ML_CONF["model_path"])
    scaler = joblib.load(ML_CONF["scaler_path"])
    print("Loaded ML model & scaler")
    return model, scaler

def backtest_ml_1h(symbols):
    model, scaler = load_ml_artifacts()
    if model is None or scaler is None:
        return pd.DataFrame(columns=["symbol","entry_time","entry_price","exit_price","pnl","bars_held","exit_reason"])

    rows = []
    for sym in symbols:
        try:
            df = load_ohlcv(sym, ML_TIMEFRAME)
        except Exception as e:
            print(f"[ml warn] {sym}: {e}")
            continue
        f = compute_features(df)
        X = f[ML_CONF["feature_columns"]].copy()
        mask = X.notna().all(axis=1)
        if mask.sum() == 0:
            continue
        X = X[mask]
        df_masked = f.loc[X.index]  # aligned to X

        Xs = scaler.transform(X.values)
        proba = model.predict_proba(Xs)[:, 1]

        # ensure 1-D integer positions from np.where
        entries = np.where(proba >= ML_CONF["proba_threshold"])[0]  # not the tuple, take [0]

        for i in entries.tolist():  # scalar ints
            # scalar timestamp in masked frame
            entry_ts = df_masked.index[i]  # a single pd.Timestamp (scalar)
            # map to integer loc in full df via searchsorted on sorted index
            entry_loc = int(df.index.searchsorted(entry_ts))
            if entry_loc >= len(df) - 1:
                continue

            entry_time = df.index[entry_loc]
            entry_price = float(df["close"].iloc[entry_loc])

            highs = df["high"].iloc[entry_loc+1:].to_numpy()
            lows  = df["low"].iloc[entry_loc+1:].to_numpy()

            tp_price = entry_price * (1 + ML_CONF["profit_target"])
            sl_price = entry_price * (1 - ML_CONF["stop_loss"])

            exit_price = entry_price
            exit_reason = "TIME"
            bars_held = 0
            for j, (h, l) in enumerate(zip(highs, lows), start=1):
                if l <= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                    bars_held = j
                    break
                if h >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    bars_held = j
                    break
                if j >= ML_CONF["max_hold_bars"]:
                    end_loc = min(entry_loc + j, len(df) - 1)
                    exit_price = float(df["close"].iloc[end_loc])
                    exit_reason = "TIME"
                    bars_held = j
                    break

            pnl = exit_price - entry_price
            rows.append([sym, entry_time, entry_price, exit_price, pnl, bars_held, exit_reason])

    cols = ["symbol","entry_time","entry_price","exit_price","pnl","bars_held","exit_reason"]
    trades = pd.DataFrame(rows, columns=cols)
    trades.to_csv(ML_TRADES_CSV, index=False)
    return trades

# -------------------------
# Aggregate summary
# -------------------------
def write_summary(rules_trades: pd.DataFrame, ml_trades: pd.DataFrame):
    rows = []
    # RULES
    rsum = summarize_trades(rules_trades)
    rows.append([
        "RULES_15M",
        rsum["total_trades"], rsum["total_pnl"], rsum["win_rate"], rsum["profit_factor"],
        rsum["avg_pnl_per_trade"], rsum["best_trade"], rsum["worst_trade"]
    ])
    # ML
    msum = summarize_trades(ml_trades)
    rows.append([
        "ML_1H",
        msum["total_trades"], msum["total_pnl"], msum["win_rate"], msum["profit_factor"],
        msum["avg_pnl_per_trade"], msum["best_trade"], msum["worst_trade"]
    ])
    cols = ["label","total_trades","total_pnl","win_rate","profit_factor",
            "avg_pnl_per_trade","best_trade","worst_trade"]
    pd.DataFrame(rows, columns=cols).to_csv(SUMMARY_CSV, index=False)
    print(f"✅ Completed. Files written:\n - {os.path.basename(RULES_TRADES_CSV)} (trades)\n - {os.path.basename(ML_TRADES_CSV)} (ML trades)\n - {os.path.basename(SUMMARY_CSV)} (aggregate metrics)")

# -------------------------
# Main
# -------------------------
def main():
    symbols = list_symbols()
    # Backtest rules (15m)
    rules_trades = backtest_rules_15m(symbols)
    # Backtest ML (1h)
    ml_trades = backtest_ml_1h(symbols)
    # Summarize
    write_summary(rules_trades, ml_trades)

if __name__ == "__main__":
    main()
