# orb_hourly_local.py
import os, math
import pandas as pd, numpy as np, pytz
from datetime import time as dtime

# ========== DATA PATHS ==========
BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily':    os.path.join(BASE_DIR, "swing_data"),
    '1hour':    os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}

TIMEFRAME = '1hour'                 # choose '1hour' for this strategy
DATA_DIR = DATA_PATHS[TIMEFRAME]

# ========== SYMBOLS & DATE FILTER ==========
START_DATE = "2024-01-01 09:15:00"
END_DATE   = "2025-09-30 15:30:00"

# ========== STRATEGY PARAMS ==========
RISK_PCT_EQUITY = 1.0               # percent of equity per trade
SLIPPAGE_POINTS = 0.10              # rupees added to stop distance
MAX_SHARES_CAP  = 100000
INITIAL_EQUITY  = 1_000_000.0

SL_TYPE   = "OppositeRange"         # OppositeRange | FixedPoints | FixedPct | ATR
SL_VALUE  = 0.0                     # points/%/ATR multiple depending on type
TP_TYPE   = "RR"                    # RR | FixedPoints | FixedPct | ATR
TP_VALUE  = 2.0                     # RR multiple or points/%/ATR mult
ATR_LEN   = 14
ALLOW_SECOND_CHANCE = True
SESSION_EOD = dtime(15,30)
IST = pytz.timezone("Asia/Kolkata")

OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

# ========== HELPERS ==========
def load_symbol_csv(data_dir: str, symbol: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    # Required columns
    req = {"datetime","open","high","low","close","volume"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    # Parse datetime as IST tz-aware
    dt = pd.to_datetime(df["datetime"])
    # If naive, localize to IST; if tz-aware but not IST, convert to IST
    if dt.dt.tz is None:
        idx = dt.tz_localize(IST)
    else:
        idx = dt.dt.tz_convert(IST)
    df = df.set_index(idx).sort_index()
    # Filter date range
    start = pd.Timestamp(START_DATE).tz_localize(IST) if pd.Timestamp(START_DATE).tzinfo is None else pd.Timestamp(START_DATE)
    end   = pd.Timestamp(END_DATE).tz_localize(IST)   if pd.Timestamp(END_DATE).tzinfo   is None else pd.Timestamp(END_DATE)
    df = df.loc[(df.index >= start) & (df.index <= end)]
    return df[["open","high","low","close","volume"]]

def compute_atr(df: pd.DataFrame, n:int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(),
                    (df["high"]-pc).abs(),
                    (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def ist_time(ts: pd.Timestamp):
    return ts.tz_convert(IST).time()

# ========== STRATEGY CORE (Hourly ORB) ==========
def backtest_symbol_hourly(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    # If data is not exactly 60-minute bars, resample to 60T
    # Assumes OHLCV at higher frequency; if already 60-minute, this preserves it
    df = (df
          .assign(v=df["volume"])
          .resample("60T", label="right", closed="right")
          .agg({"open":"first","high":"max","low":"min","close":"last","v":"sum"})
          .rename(columns={"v":"volume"})
          .dropna())
    df = df.sort_index()
    df["atr"] = compute_atr(df, ATR_LEN)

    equity = INITIAL_EQUITY
    in_pos = False
    qty = 0
    entry = np.nan
    last_ref_high = np.nan
    last_ref_low = np.nan
    reentry_used = False
    trades = []

    prev = None
    for ts, row in df.iterrows():
        # EOD square-off
        if in_pos and ist_time(ts) >= SESSION_EOD:
            exit_px = row["close"]
            pnl = (exit_px - entry) * qty
            equity += pnl
            trades.append(dict(ts=ts, symbol=symbol, side="EOD_CLOSE",
                               entry=entry, exit=exit_px, qty=qty, pnl=pnl, equity=equity))
            in_pos = False; qty=0; entry=np.nan

        if prev is None:
            prev = row
            continue

        ref_high = float(prev["high"])
        ref_low  = float(prev["low"])
        last_ref_high, last_ref_low = ref_high, ref_low

        price = float(row["close"])
        atr   = float(row["atr"]) if not math.isnan(row["atr"]) else None

        if in_pos:
            # Build SL
            if SL_TYPE == "OppositeRange":
                stop = last_ref_low
            elif SL_TYPE == "FixedPoints":
                stop = entry - SL_VALUE
            elif SL_TYPE == "FixedPct":
                stop = entry * (1.0 - SL_VALUE/100.0)
            elif SL_TYPE == "ATR" and atr is not None:
                stop = entry - SL_VALUE * atr
            else:
                stop = last_ref_low
            risk = abs(entry - stop)
            # Build TP
            if TP_TYPE == "RR":
                target = entry + TP_VALUE * risk
            elif TP_TYPE == "FixedPoints":
                target = entry + TP_VALUE
            elif TP_TYPE == "FixedPct":
                target = entry * (1.0 + TP_VALUE/100.0)
            elif TP_TYPE == "ATR" and atr is not None:
                target = entry + TP_VALUE * atr
            else:
                target = entry + 2.0 * risk

            exit_reason = None; exit_px = None
            if price <= stop:
                exit_reason = "SL"; exit_px = stop
            elif price >= target:
                exit_reason = "TP"; exit_px = target

            if exit_reason:
                pnl = (exit_px - entry) * qty
                equity += pnl
                trades.append(dict(ts=ts, symbol=symbol, side=exit_reason,
                                   entry=entry, exit=exit_px, qty=qty, pnl=pnl, equity=equity))
                in_pos = False; qty=0; entry=np.nan
                if exit_reason == "SL":
                    reentry_used = False
                else:
                    reentry_used = True

        if not in_pos:
            breakout = price > ref_high
            can_enter = breakout and (ALLOW_SECOND_CHANCE or not reentry_used)
            if can_enter:
                # Sizing
                if SL_TYPE == "OppositeRange":
                    stop_for_size = ref_low
                elif SL_TYPE == "FixedPoints":
                    stop_for_size = price - SL_VALUE
                elif SL_TYPE == "FixedPct":
                    stop_for_size = price * (1.0 - SL_VALUE/100.0)
                elif SL_TYPE == "ATR" and atr is not None:
                    stop_for_size = price - SL_VALUE * atr
                else:
                    stop_for_size = ref_low
                stop_dist = max(0.01, abs(price - stop_for_size) + SLIPPAGE_POINTS)
                risk_amt = equity * (RISK_PCT_EQUITY/100.0)
                q = int(max(1, min(MAX_SHARES_CAP, math.floor(risk_amt / stop_dist))))
                if q > 0:
                    in_pos = True
                    qty = q
                    entry = ref_high
                    trades.append(dict(ts=ts, symbol=symbol, side="BUY",
                                       entry=entry, exit=np.nan, qty=qty, pnl=0.0, equity=equity))
                    # If second-chance allowed, keep reentry_used False after SL only.
                    # Set reentry_used True here to prevent same-reference repeated entries.
                    reentry_used = True

        prev = row

    return pd.DataFrame(trades)

def main():
    all_trades = []
    for sym in SYMBOLS:
        df = load_symbol_csv(DATA_DIR, sym)
        if df.empty:
            print(f"No data for {sym}")
            continue
        trades = backtest_symbol_hourly(df, sym)
        trades["symbol_file"] = sym
        all_trades.append(trades)

    if not all_trades:
        print("No trades.")
        return

    trades = pd.concat(all_trades).sort_values("ts")
    trades.to_csv(os.path.join(OUT_DIR, "trades_hourly_orb_local.csv"), index=False)

    # Build equity curve using last recorded equity per trade chronologically
    curve = trades[["ts","equity"]].dropna().drop_duplicates("ts")
    curve.to_csv(os.path.join(OUT_DIR, "equity_curve_local.csv"), index=False)

    print("Saved: out/trades_hourly_orb_local.csv and out/equity_curve_local.csv")

if __name__ == "__main__":
    main()
