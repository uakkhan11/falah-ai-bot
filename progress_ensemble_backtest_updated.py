#!/usr/bin/env python3
import os
import sys
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Config & CLI
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = BASE_DIR  # write outputs in project root

DEFAULT_FEATURES = ["ret", "rsi", "sma"]  # must match harness ML_CONF.feature_columns
MODEL_PATH = os.path.join(MODELS_DIR, "best_rf_1h.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "best_scaler_1h.pkl")

def parse_args():
    p = argparse.ArgumentParser(description="Train ML ensemble (1h) and export artifacts + signals.")
    p.add_argument("--quick", action="store_true", help="Limit symbols for a fast validation pass")  # boolean flag [8]
    p.add_argument("--symbols", nargs="*", help="Explicit list of symbols to include")  # multiple values [6][10]
    p.add_argument("--timeframes", nargs="*", default=["1h"], help="Timeframes to evaluate (default: 1h)")
    p.add_argument("--threshold", type=float, default=0.65, help="Confidence threshold for entries")
    p.add_argument("--estimators", type=int, default=50, help="RandomForest n_estimators")
    p.add_argument("--max_samples", type=int, default=None, help="Optional cap on samples per symbol for speed")
    return p.parse_args()

# -------------------------
# Data loading stubs
# -------------------------
def list_symbols():
    # Replace with actual symbol discovery; keeping a deterministic subset if quick mode requested elsewhere
    # This function returns a large consolidated list; filter later with args
    return sorted(list(set([
        # Fill with exchange universe or read from a symbols file; examples below align with attachments
        "KANSAINER","BERGEPAINT","RHIM","WABAG","CONCORDBIO","AXISCADES-BE","LUMAXTECH","POWERINDIA","SHARDACROP",
        "GALLANTT","KIOCL","SYRMA","TARIL","TRANSRAILL","WEBELSOLAR","FORCEMOT","POKARNA","THYROCARE","YATHARTH",
        "SKIPPER","PRIVISCL","SHARDAMOTR","GABRIEL","PTCIL","NETWEB","LLOYDSENGG","BANCOINDIA","SHANTIGEAR","TDPOWERSYS",
        "CEWATER","WAAREEENER","SUNFLAG","WOCKPHARMA","EMAMILTD","AVANTIFEED","LUXIND","FINPIPE","RAINBOW",
        "BHARTIARTL","HINDALCO","TORNTPHARM","ULTRACEMCO","OIL","JUBLFOOD","VOLTAS"
    ])))

def load_ohlcv(symbol: str, timeframe: str):
    # Placeholder: replace with actual OHLCV loader returning DataFrame with columns: ['open','high','low','close','volume']
    # Ensure index is datetime-like ascending
    # For example: pd.read_parquet(f"{DATA_DIR}/{timeframe}/{symbol}.parquet")
    raise NotImplementedError("Implement load_ohlcv(symbol, timeframe) to return OHLCV DataFrame.")

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Compute ret, rsi, sma; simple versions for alignment with DEFAULT_FEATURES
    out = df.copy()
    out["ret"] = out["close"].pct_change().fillna(0.0)
    # RSI(14) simple calc
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

def label_targets(df: pd.DataFrame) -> pd.Series:
    # Simple forward-return > 0 labeling at horizon H=3 bars
    H = 3
    fwd_ret = df["close"].shift(-H) / df["close"] - 1.0
    y = (fwd_ret > 0).astype(int)
    return y.fillna(0)

# -------------------------
# Training & caching
# -------------------------
def train_global_model(symbols, timeframe, feature_cols, n_estimators=50, max_samples=None):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Use cached artifacts if present
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("✅ Using cached ML model/scaler from models/")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler

    X_list, y_list = [], []
    for sym in symbols:
        try:
            df = load_ohlcv(sym, timeframe)
        except Exception as e:
            print(f"[warn] {sym}: load failed: {e}")
            continue
        df = compute_features(df)
        y = label_targets(df)
        X = df[feature_cols].copy()
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        if max_samples and len(X) > max_samples:
            X = X.tail(max_samples)
            y = y.tail(max_samples)
        if len(X) < 100:
            continue
        X_list.append(X.values)
        y_list.append(y.values)

    if not X_list:
        raise RuntimeError("No training data aggregated; check loaders and features.")

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)  # save the scaler for inference [9]

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)  # [3]
    model.fit(X_scaled, y_all)

    # Persist artifacts [2][5]
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Saved model and scaler to {MODELS_DIR}")
    return model, scaler

# -------------------------
# Inference & signal export
# -------------------------
def generate_signals(symbols, timeframe, feature_cols, model, scaler, proba_threshold=0.65):
    rows = []
    per_symbol_rows = []
    for sym in symbols:
        try:
            df = load_ohlcv(sym, timeframe)
        except Exception as e:
            print(f"[warn] {sym}: load failed: {e}")
            continue
        df = compute_features(df)
        X = df[feature_cols].copy()
        mask = X.notna().all(axis=1)
        X = X[mask]
        if len(X) == 0:
            continue
        Xs = scaler.transform(X.values)
        proba = model.predict_proba(Xs)[:, 1]
        entries = np.where(proba >= proba_threshold)
        for idx in entries:
            # Map back to original index position
            entry_pos = X.index[idx]
            entry_idx = df.index.get_loc(entry_pos)
            entry_price = df.loc[entry_pos, "close"]
            conf = float(proba[idx])
            rows.append((sym, entry_idx, float(entry_price), conf))
        # Optional: simple PnL proxy for reporting (aligns roughly with attachments)
        # This is placeholder; real PnL is computed in harness. Here we only summarize counts.
        if len(entries) > 0:
            per_symbol_rows.append((sym, len(entries), float(np.mean(proba[entries]))))
    # Write ensemble signals file in the exact format seen in attachments
    out_path = os.path.join(OUT_DIR, "ensemble_trades_updated.txt")
    with open(out_path, "w") as f:
        f.write("symbol        entry_idx  entry_price  confidence\n")
        for sym, eidx, price, conf in rows:
            f.write(f"{sym:<13}{eidx:<12}{price:<12.2f}{conf:.2f}\n")
    print(f"📝 Wrote {len(rows)} signals to {out_path}")

    return rows, per_symbol_rows

# -------------------------
# Reporting (headers to match prior CSVs)
# -------------------------
def write_placeholder_aggregates():
    # These are headers; actual aggregated PnL is produced by harness. Kept for downstream tooling compatibility.
    tf_path = os.path.join(OUT_DIR, "timeframe_performance_analysis.csv")
    with open(tf_path, "w") as f:
        f.write("timeframe,Trades,Total_PnL,Avg_PnL,Win_Rate,Avg_Confidence,Profit_Target_%\n")
    mo_path = os.path.join(OUT_DIR, "monthly_performance_analysis.csv")
    with open(mo_path, "w") as f:
        f.write("month,Trades,Total_PnL,Win_Rate\n")
    print(f"📄 Created headers: timeframe_performance_analysis.csv, monthly_performance_analysis.csv")

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    all_syms = list_symbols()

    # Symbol selection
    if args.symbols and len(args.symbols) > 0:
        symbols = [s for s in args.symbols if s in all_syms] or args.symbols
    else:
        symbols = all_syms

    # Quick mode limits for fast validation
    if args.quick:
        # Prioritize high-conviction core first (subset)
        priority = [
            "AXISCADES-BE","LUMAXTECH","POWERINDIA","SHARDACROP","GALLANTT","KIOCL",
            "SYRMA","TARIL","TRANSRAILL","WEBELSOLAR","FORCEMOT","POKARNA","THYROCARE",
            "YATHARTH","SKIPPER","PRIVISCL","SHARDAMOTR","GABRIEL","PTCIL","NETWEB",
            # Include coverage tier from ensemble attachments
            "KANSAINER","BERGEPAINT","RHIM","WABAG","CONCORDBIO"
        ]
        symbols = [s for s in priority if s in symbols]
        if not symbols:
            symbols = all_syms[:25]

    tfs = args.timeframes
    feature_cols = DEFAULT_FEATURES
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Train or load cached artifacts for 1h only (primary)
    primary_tf = "1h"
    if primary_tf not in tfs:
        tfs = [primary_tf] + [tf for tf in tfs if tf != primary_tf]

    print(f"Symbols: {len(symbols)} | Timeframes: {tfs} | Threshold: {args.threshold}")
    model, scaler = train_global_model(
        symbols=symbols,
        timeframe=primary_tf,
        feature_cols=feature_cols,
        n_estimators=args.estimators,
        max_samples=args.max_samples
    )

    # Generate signals for requested timeframes (1h at minimum)
    for tf in tfs:
        try:
            rows, per_sym = generate_signals(symbols, tf, feature_cols, model, scaler, args.threshold)
            print(f"[{tf}] signals: {len(rows)} | symbols with entries: {len(per_sym)}")
        except NotImplementedError:
            print(f"[skip] Data loader not implemented for timeframe={tf}.")
            continue

    # Write placeholder aggregate CSV headers (real PnL via harness)
    write_placeholder_aggregates()
    print("✅ Completed. Artifacts in models/, signals in ensemble_trades_updated.txt")

if __name__ == "__main__":
    main()
