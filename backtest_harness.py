import os, json, warnings
import numpy as np, pandas as pd
import ta, joblib
warnings.filterwarnings("ignore")

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    "15minute": os.path.join(BASE_DIR, "scalping_data"),
    "1hour": os.path.join(BASE_DIR, "intraday_swing_data"),
    "daily": os.path.join(BASE_DIR, "swing_data"),
}
YEAR_FILTER = 2025

# --- Rules strategy parameters (15m baseline) ---
RULES = {
    "RSI_OVERBOUGHT": 70,
    "ADX_MIN": 20,
    "VOLUME_MULT": 1.5,
    "SMA_PERIOD": 20,
    "PROFIT_TARGET": 0.025,
    "STOP_LOSS": 0.015,
    "MAX_HOLD_BARS": 25,
}

# --- ML strategy parameters (1h) ---
ML_CONF = {
    "model_path": os.path.join(BASE_DIR, "models", "best_rf_1h.pkl"),
    "scaler_path": os.path.join(BASE_DIR, "models", "best_scaler_1h.pkl"),
    "feature_columns": ["returns", "rsi", "bb_position", "bb_width", "adx", "momentum_5", "roc_5"],
    "proba_threshold": 0.65,
    "profit_target": 0.035,
    "stop_loss": 0.020,
    "max_hold_bars": 12,
}

def list_symbols(source="daily", limit=None):
    path = DATA_PATHS[source]
    files = [f for f in os.listdir(path) if f.endswith(".csv")] if os.path.exists(path) else []
    syms = sorted([f[:-4] for f in files])
    return syms if limit is None else syms[:limit]

def load_df(symbol, source):
    p = os.path.join(DATA_PATHS[source], f"{symbol}.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    if "date" not in df.columns: return None
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year == YEAR_FILTER].sort_values("date").reset_index(drop=True)
    return df

# -------- Indicators & Signals (shared with live) --------
def add_base_indicators(df, sma_period=20, rsi_p=14, adx_p=14):
    out = df.copy()
    out["rsi"] = ta.momentum.rsi(out["close"], window=rsi_p)
    out["adx"] = ta.trend.adx(out["high"], out["low"], out["close"], window=adx_p)
    out["volume_sma"] = out["volume"].rolling(sma_period).mean()
    out["volume_ratio"] = out["volume"] / out["volume_sma"]
    out["sma_20"] = out["close"].rolling(sma_period).mean()
    out["high_20"] = out["high"].rolling(sma_period).max()
    out = out.fillna(method="ffill").fillna(method="bfill")
    return out

def four_of_six_signals(df, params):
    p = params
    out = add_base_indicators(df, sma_period=p["SMA_PERIOD"])
    out["signal"] = 0
    start = max(50, len(out)//4)
    for i in range(start, len(out)):
        cur, prev = out.iloc[i], out.iloc[i-1]
        conds = [
            cur["close"] > cur["high_20"],
            cur["volume_ratio"] > p["VOLUME_MULT"],
            cur["adx"] > p["ADX_MIN"],
            cur["rsi"] < p["RSI_OVERBOUGHT"],
            cur["close"] > cur["sma_20"],
            cur["rsi"] > prev["rsi"],
        ]
        if sum(conds) >= 4:
            out.iat[i, out.columns.get_loc("signal")] = 1
    return out

def add_ml_features(df, lookback=20):
    f = df.copy()
    f["returns"] = f["close"].pct_change().fillna(0)
    f["rsi"] = ta.momentum.rsi(f["close"], window=14)
    bb_u = ta.volatility.bollinger_hband(f["close"])
    bb_l = ta.volatility.bollinger_lband(f["close"])
    bb_m = ta.volatility.bollinger_mavg(f["close"])
    f["bb_position"] = (f["close"] - bb_l) / (bb_u - bb_l)
    f["bb_width"] = (bb_u - bb_l) / bb_m
    f["adx"] = ta.trend.adx(f["high"], f["low"], f["close"], window=14)
    for period in [12]:
        f[f"momentum_{period}"] = f["close"] / f["close"].shift(period)
        f[f"roc_{period}"] = ((f["close"] - f["close"].shift(period))/f["close"].shift(period))*100
    # clean
    num_cols = f.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        f[c] = f[c].replace([np.inf, -np.inf], np.nan)
        f[c] = f[c].fillna(method="ffill").fillna(method="bfill")
        f[c] = f[c].fillna(f[c].median())
    return f

def ml_proba_series(df, model, scaler, feature_columns, proba_threshold):
    feats = add_ml_features(df)
    sig = np.zeros(len(feats), dtype=int)
    start = 25
    for i in range(start, len(feats)):
        x = feats.iloc[i][feature_columns].values.reshape(1, -1)
        if np.isnan(x).any() or np.isinf(x).any(): continue
        x_scaled = scaler.transform(x)
        proba = model.predict_proba(x_scaled)[13]
        sig[i] = 1 if proba >= proba_threshold else 0
    return sig

# -------- Backtest Engines --------
def backtest_rules_15m(symbol):
    df15 = load_df(symbol, "15minute")
    if df15 is None or len(df15) < 200: return None, None
    dfd = load_df(symbol, "daily")
    if dfd is None or len(dfd) < 50: return None, None
    dfd = add_base_indicators(dfd, sma_period=RULES["SMA_PERIOD"])
    df15 = four_of_six_signals(df15, RULES)
    # simple daily trend filter: close above SMA
    daily_up = dfd.iloc[-1]["close"] > dfd.iloc[-1]["sma_20"]

    cash = 100000
    positions = {}
    trades = []
    for i in range(1, len(df15)):
        cur = df15.iloc[i]
        # exits
        to_close = []
        for pid, pos in positions.items():
            ret = (cur["close"] - pos["entry_price"]) / pos["entry_price"]
            bars = i - pos["entry_bar"]
            exit_reason = None
            if ret >= RULES["PROFIT_TARGET"]:
                exit_reason = "Profit Target"
            elif ret <= -RULES["STOP_LOSS"]:
                exit_reason = "Stop Loss"
            elif bars >= RULES["MAX_HOLD_BARS"]:
                exit_reason = "Time Exit"
            if exit_reason:
                shares = pos["shares"]
                exit_price = cur["close"] * 0.9995
                pnl = (exit_price - pos["entry_price"]) * shares
                commission = (pos["entry_price"] + exit_price) * shares * 0.0005
                net = pnl - commission
                trades.append({
                    "symbol": symbol, "entry_date": pos["entry_date"], "exit_date": cur["date"],
                    "entry_price": pos["entry_price"], "exit_price": exit_price, "shares": shares,
                    "pnl": net, "return_pct": ret, "exit_reason": exit_reason, "bars_held": bars
                })
                cash += pos["entry_value"] + net
                to_close.append(pid)
        for pid in to_close: del positions[pid]

        # entries
        if cur.get("signal", 0) == 1 and daily_up and len(positions) < 3 and cash > 5000:
            position_value = cash * 0.02
            entry_price = cur["close"] * 1.0005
            shares = position_value / entry_price
            if shares > 0:
                positions[len(positions)] = {
                    "entry_date": cur["date"], "entry_price": entry_price,
                    "shares": shares, "entry_bar": i, "entry_value": position_value
                }
                cash -= position_value
    return pd.DataFrame(trades), {"cash_final": cash}

def backtest_ml_1h(symbol, bundle):
    df1h = load_df(symbol, "1hour")
    if df1h is None or len(df1h) < 100: return None, None
    sig = ml_proba_series(df1h, bundle["model"], bundle["scaler"], bundle["feature_columns"], bundle["proba_threshold"])
    cash = 100000
    pos = {}
    trades = []
    for i in range(1, len(df1h)):
        cur = df1h.iloc[i]
        # exit
        to_close = []
        for pid, p in pos.items():
            ret = (cur["close"] - p["entry_price"]) / p["entry_price"]
            bars = i - p["entry_bar"]
            exit_reason = None
            if ret >= ML_CONF["profit_target"]:
                exit_reason = "Profit Target"
            elif ret <= -ML_CONF["stop_loss"]:
                exit_reason = "Stop Loss"
            elif bars >= ML_CONF["max_hold_bars"]:
                exit_reason = "Time Exit"
            if exit_reason:
                shares = p["shares"]
                exit_price = cur["close"] * 0.999
                pnl = (exit_price - p["entry_price"]) * shares
                commission = (p["entry_price"] + exit_price) * shares * 0.0003
                net = pnl - commission
                trades.append({
                    "symbol": symbol, "entry_date": p["entry_date"], "exit_date": cur["date"],
                    "entry_price": p["entry_price"], "exit_price": exit_price, "shares": shares,
                    "pnl": net, "return_pct": ret, "exit_reason": exit_reason, "bars_held": bars
                })
                cash += p["entry_value"] + net
                to_close.append(pid)
        for pid in to_close: del pos[pid]
        # entry
        if len(pos)==0 and cash>10000 and sig[i]==1:
            position_value = cash * 0.02
            entry_price = cur["close"] * 1.001
            shares = position_value / entry_price
            if shares>0:
                pos = {
                    "entry_date": cur["date"], "entry_price": entry_price,
                    "shares": shares, "entry_bar": i, "entry_value": position_value
                }
                cash -= position_value
    return pd.DataFrame(trades), {"cash_final": cash}

def summarize(trades_df, label):
    if trades_df is None or len(trades_df)==0:
        return {"label": label, "total_trades": 0, "total_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    wins = trades_df[trades_df["pnl"]>0]
    losses = trades_df[trades_df["pnl"]<=0]
    pf = abs(wins["pnl"].sum()/losses["pnl"].sum()) if len(losses)>0 else np.inf
    wr = len(wins)/len(trades_df)*100
    return {
        "label": label,
        "total_trades": int(len(trades_df)),
        "total_pnl": float(trades_df["pnl"].sum()),
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "avg_pnl_per_trade": float(trades_df["pnl"].mean()),
        "best_trade": float(trades_df["pnl"].max()),
        "worst_trade": float(trades_df["pnl"].min()),
    }

if __name__ == "__main__":
    print("🚀 Unified Backtest: 15m Rules vs 1h ML")

    symbols = list_symbols("daily")  # master list from daily directory
    print(f"Symbols discovered: {len(symbols)}")

    # load ML artifacts if present
    ml_bundle = None
    if os.path.exists(ML_CONF["model_path"]) and os.path.exists(ML_CONF["scaler_path"]):
        ml_bundle = {
            "model": joblib.load(ML_CONF["model_path"]),
            "scaler": joblib.load(ML_CONF["scaler_path"]),
            "feature_columns": ML_CONF["feature_columns"],
            "proba_threshold": ML_CONF["proba_threshold"],
        }
        print("Loaded ML model & scaler")

    results_rules = []
    results_ml = []

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}")
        # Rules 15m
        tr15, meta15 = backtest_rules_15m(sym)
        if tr15 is not None and len(tr15)>0:
            tr15["strategy"] = "RULES_15M"
            results_rules.append(tr15)
        # ML 1h
        if ml_bundle:
            trml, metaml = backtest_ml_1h(sym, ml_bundle)
            if trml is not None and len(trml)>0:
                trml["strategy"] = "ML_1H"
                results_ml.append(trml)

    df_rules = pd.concat(results_rules, ignore_index=True) if results_rules else pd.DataFrame()
    df_ml = pd.concat(results_ml, ignore_index=True) if results_ml else pd.DataFrame()

    # Save detailed trades
    if len(df_rules)>0:
        df_rules.to_csv("results_rules_15m.csv", index=False)
    if len(df_ml)>0:
        df_ml.to_csv("results_ml_1h.csv", index=False)

    # Summaries
    sum_rules = summarize(df_rules, "RULES_15M")
    sum_ml = summarize(df_ml, "ML_1H") if len(df_ml)>0 else summarize(None,"ML_1H")
    summary_df = pd.DataFrame([sum_rules, sum_ml])
    summary_df.to_csv("summary_compare.csv", index=False)

    print("\n✅ Completed. Files written:")
    print(" - results_rules_15m.csv (trades)") if len(df_rules)>0 else print(" - No RULES trades")
    print(" - results_ml_1h.csv (trades)") if len(df_ml)>0 else print(" - No ML trades or ML not loaded")
    print(" - summary_compare.csv (aggregate metrics)")
