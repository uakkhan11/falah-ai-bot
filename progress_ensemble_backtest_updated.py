#!/usr/bin/env python3
import os, sys, json, math, itertools, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# ---------------- Config: freeze dataset ----------------
BASE_DIR = "/root/falah-ai-bot"
DATA_DIRS = {
    "15m": os.path.join(BASE_DIR, "scalping_data"),
    "1h":  os.path.join(BASE_DIR, "intraday_swing_data"),
    "1d":  os.path.join(BASE_DIR, "swing_data"),
}
# Freeze a universe (edit to your 20 names present on disk)
SYMBOLS = [
    "AVANTIFEED","AXISCADES-BE","BANCOINDIA","BERGEPAINT","BHARTIARTL",
    "CEWATER","CONCORDBIO","EMAMILTD","FINPIPE","FORCEMOT",
    "GABRIEL","GALLANTT","HINDALCO","JUBLFOOD","KANSAINER",
    "KIOCL","LLOYDSENGG","LUMAXTECH","LUXIND","OIL"
]

DATE_START = "2024-01-01"
DATE_END   = "2025-06-30"

# Walk-forward rolling windows: 6 months train, 1 month test
TRAIN_MONTHS = 6
TEST_MONTHS  = 1

# Parameter grid (keep small and fixed)
PROBA_GRID = [0.70, 0.75, 0.80]
PT_GRID    = [0.04, 0.05]     # profit target
SL_GRID    = [0.008, 0.01]    # stop loss
HOLD_GRID  = [24, 30]         # max hold bars on 1h

# Output
OUT_DIR = os.path.join(BASE_DIR, "wf_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Data & features ----------------
def load_csv(dirpath, symbol):
    path = os.path.join(dirpath, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # handle common timestamp column names
    for c in ["timestamp","date","Datetime","Date","time"]:
        if c in df.columns:
            df["date"] = pd.to_datetime(df[c])
            break
    if "date" not in df:
        raise ValueError(f"{symbol}: no timestamp column")
    cols = {c.lower(): c for c in df.columns}
    # normalize
    need = ["open","high","low","close","volume"]
    out = pd.DataFrame({
        "date": df["date"].values,
        "open": df[cols.get("open","open")].astype(float).values,
        "high": df[cols.get("high","high")].astype(float).values,
        "low":  df[cols.get("low","low")].astype(float).values,
        "close":df[cols.get("close","close")].astype(float).values,
        "volume":df[cols.get("volume","volume")].astype(float).values,
    })
    out = out.sort_values("date").reset_index(drop=True)
    return out

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def atr(df, window=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr1 = (h - l).abs()
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def adx(df, window=14):
    # simplified ADX computation
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = atr(df, window=1) * 1.0
    atrn = tr.rolling(window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window).sum() / (atrn + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window).sum() / (atrn + 1e-9)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    return dx.rolling(window).mean()

def compute_features(df_15m, df_1h):
    # regime/trend context from 1h
    df_1h = df_1h.copy()
    df_1h["ema200"] = ema(df_1h["close"], 200)
    df_1h["adx14"]  = adx(df_1h, 14)
    df_1h["atr14"]  = atr(df_1h, 14)
    df_1h["ema50"]  = ema(df_1h["close"], 50)
    df_1h["ema200_slope"] = df_1h["ema200"].diff()

    # map 1h to 15m by ffill on timestamp
    h = df_1h.set_index("date")
    m = df_15m.set_index("date")
    for col in ["ema200","adx14","atr14","ema50","ema200_slope","close"]:
        m[f"h_{col}"] = h[col].reindex(m.index, method="ffill")

    m["ret"] = m["close"].pct_change().fillna(0.0)
    m["atr14_norm"] = (m["h_atr14"] / m["h_close"]).fillna(0.0)
    m["regime"] = ((m["h_close"] > m["h_ema200"]) & (m["h_adx14"] > 20)).astype(int)
    # compact feature set
    feat_cols = ["ret","atr14_norm","h_adx14","h_ema200_slope"]
    m = m.reset_index()
    return m, feat_cols

# ---------------- Labeling & ML ----------------
def make_labels(m15, horizon_bars=10, thresh=0.01):
    # Future return over horizon on 15m data
    fut = m15["close"].shift(-horizon_bars) / m15["close"] - 1
    m15["label"] = (fut > thresh).astype(int)
    return m15

def fit_predict_prob(train_df, test_df, feat_cols):
    # Strict time hygiene: fit scaler & model only on train
    Xtr = train_df[feat_cols].values
    ytr = train_df["label"].values
    Xte = test_df[feat_cols].values
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4
    )
    model.fit(Xtr_s, ytr)
    proba = model.predict_proba(Xte_s)[:, 1]
    return proba

# ---------------- Backtest (1h exit logic applied to 15m entries by time) ----------------
def backtest_entries(df, proba, proba_thresh, pt, sl, max_hold_bars):
    df = df.copy().reset_index(drop=True)
    df["proba"] = proba
    df["ml_entry"] = ((df["proba"] >= proba_thresh) & (df["regime"] == 1)).astype(int)

    # Convert 15m to 1h exit granularity by sampling every 4 bars for highs/lows window
    rows = []
    pos_open = None
    for i in range(len(df)-1):
        if pos_open is None and df.loc[i, "ml_entry"] == 1:
            pos_open = {
                "entry_idx": i,
                "entry_time": df.loc[i, "date"],
                "entry_price": float(df.loc[i, "close"])
            }
            continue
        if pos_open is not None:
            eidx = pos_open["entry_idx"]
            entry_price = pos_open["entry_price"]
            # build pseudo-1h windows
            held = i - eidx
            if held <= 0:
                continue
            # price path since entry
            hslice = df.loc[eidx+1:i, "high"]
            lslice = df.loc[eidx+1:i, "low"]
            tp = entry_price * (1 + pt)
            slp = entry_price * (1 - sl)
            hit = None
            # evaluate bar by bar
            for j, (h, l) in enumerate(zip(hslice.values, lslice.values), start=1):
                if l <= slp:
                    rows.append((pos_open["entry_time"], entry_price, slp, - (entry_price - slp), j, "SL"))
                    pos_open = None
                    hit = True
                    break
                if h >= tp:
                    rows.append((pos_open["entry_time"], entry_price, tp, tp - entry_price, j, "TP"))
                    pos_open = None
                    hit = True
                    break
                if j >= max_hold_bars:
                    exit_price = float(df.loc[eidx + j, "close"])
                    pnl = exit_price - entry_price
                    rows.append((pos_open["entry_time"], entry_price, exit_price, pnl, j, "TIME"))
                    pos_open = None
                    hit = True
                    break
            if hit:
                continue
    # finalize
    trades = pd.DataFrame(rows, columns=["entry_time","entry_price","exit_price","pnl","bars_held","exit_reason"])
    return trades

def summarize_trades(trades):
    if trades.empty:
        return {"total_trades":0,"total_pnl":0.0,"win_rate":0.0,"profit_factor":0.0,"avg_pnl":0.0}
    total = len(trades)
    total_pnl = trades["pnl"].sum()
    wins = (trades["pnl"] > 0).sum()
    win_rate = 100.0 * wins / total
    gp = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gl = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    pf = (gp / gl) if gl > 0 else (gp if gp > 0 else 0.0)
    avg = total_pnl / total
    return {"total_trades":total,"total_pnl":float(total_pnl),"win_rate":float(win_rate),
            "profit_factor":float(pf),"avg_pnl":float(avg)}

# ---------------- Walk-forward runner ----------------
def month_floor(d):
    return pd.Timestamp(d).to_period("M").to_timestamp()

def generate_folds(df_15m):
    # Monthly buckets
    df = df_15m[(df_15m["date"] >= DATE_START) & (df_15m["date"] <= DATE_END)].copy()
    df["ym"] = df["date"].dt.to_period("M")
    months = sorted(df["ym"].unique())
    folds = []
    for i in range(TRAIN_MONTHS, len(months) - TEST_MONTHS + 1):
        train_months = months[i-TRAIN_MONTHS:i]
        test_months = months[i:i+TEST_MONTHS]
        tr = df[df["ym"].isin(train_months)].copy()
        te = df[df["ym"].isin(test_months)].copy()
        if len(tr) == 0 or len(te) == 0:
            continue
        folds.append((tr, te, str(test_months[0])))
    return folds

def run_symbol(symbol, grid):
    # Load 15m & 1h
    try:
        m15 = load_csv(DATA_DIRS["15m"], symbol)
        h1  = load_csv(DATA_DIRS["1h"], symbol)
    except Exception as e:
        print(f"[data] {symbol}: {e}")
        return []

    # Compute features & labels
    m15_feat, feat_cols = compute_features(m15, h1)
    m15_feat = make_labels(m15_feat, horizon_bars=10, thresh=0.01)
    m15_feat = m15_feat.dropna(subset=feat_cols + ["label"]).reset_index(drop=True)

    folds = generate_folds(m15_feat)
    results = []

    for (proba_th, pt, sl, hold) in grid:
        label = f"p{int(proba_th*100)}_pt{int(pt*100)}_sl{int(sl*1000)}_hold{hold}"
        all_trades = []
        fold_rows = []
        for tr, te, test_month in folds:
            # Fit model on train, proba on test
            proba = fit_predict_prob(tr, te, feat_cols)
            trades = backtest_entries(te, proba, proba_th, pt, sl, hold)
            trades["symbol"] = symbol
            trades["test_month"] = test_month
            all_trades.append(trades)
            met = summarize_trades(trades)
            fold_rows.append({
                "symbol": symbol, "label": label, "test_month": test_month,
                "total_trades": met["total_trades"], "total_pnl": met["total_pnl"],
                "win_rate": met["win_rate"], "profit_factor": met["profit_factor"], "avg_pnl": met["avg_pnl"]
            })
        if len(all_trades) == 0:
            continue
        trades_df = pd.concat(all_trades, ignore_index=True)
        metrics_df = pd.DataFrame(fold_rows)
        # Overall out-of-sample metrics for this symbol+combo
        overall = summarize_trades(trades_df)
        overall_row = {
            "symbol": symbol, "label": label, "test_month": "OVERALL",
            "total_trades": overall["total_trades"], "total_pnl": overall["total_pnl"],
            "win_rate": overall["win_rate"], "profit_factor": overall["profit_factor"], "avg_pnl": overall["avg_pnl"]
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([overall_row])], ignore_index=True)

        trades_out = os.path.join(OUT_DIR, f"wf_trades_{symbol}_{label}.csv")
        metrics_out = os.path.join(OUT_DIR, f"wf_metrics_{symbol}_{label}.csv")
        trades_df.to_csv(trades_out, index=False)
        metrics_df.to_csv(metrics_out, index=False)
        results.append((label, trades_df, metrics_df))
    return results

def main():
    grid = list(itertools.product(PROBA_GRID, PT_GRID, SL_GRID, HOLD_GRID))
    summary_rows = []
    for sym in SYMBOLS:
        res = run_symbol(sym, grid)
        for (label, trades_df, metrics_df) in res:
            # pull overall row for symbol+label
            overall = metrics_df[metrics_df["test_month"] == "OVERALL"].iloc[0]
            summary_rows.append({
                "symbol": sym, "label": label,
                "total_trades": overall["total_trades"],
                "total_pnl": overall["total_pnl"],
                "win_rate": overall["win_rate"],
                "profit_factor": overall["profit_factor"],
                "avg_pnl": overall["avg_pnl"]
            })
    if len(summary_rows) == 0:
        print("No outputs generated. Check data availability and date filters.")
        return
    summary_df = pd.DataFrame(summary_rows)
    # Aggregate across symbols per label to rank combos
    agg = summary_df.groupby("label").agg(
        total_trades=("total_trades","sum"),
        total_pnl=("total_pnl","sum"),
        avg_pf=("profit_factor","mean"),
        avg_pnl=("avg_pnl","mean")
    ).reset_index().sort_values(["avg_pf","avg_pnl","total_trades"], ascending=[False, False, False])
    summary_path = os.path.join(OUT_DIR, "wf_summary.csv")
    agg.to_csv(summary_path, index=False)
    print(f"✅ Done. Per-symbol files in {OUT_DIR}. Overall ranking written to wf_summary.csv")
for sym in SYMBOLS:
    try:
        m15 = load_csv(DATA_DIRS["15m"], sym)
        h1  = load_csv(DATA_DIRS["1h"], sym)
        print(sym, "15m rows:", len(m15), "range:", m15["date"].min(), "→", m15["date"].max())
        print(sym, "1h  rows:", len(h1),  "range:", h1["date"].min(),  "→", h1["date"].max())
    except Exception as e:
        print("[data]", sym, e)

if __name__ == "__main__":
    main()
