# orb_hourly_kite.py
import os, math, time as _time, datetime as dt
import pandas as pd, numpy as np, pytz
from dateutil import tz
from kiteconnect import KiteConnect

# ========== CONFIG ==========
API_KEY     = os.getenv("KITE_API_KEY")     or "ijzeuwuylr3g0kug"
API_SECRET  = os.getenv("KITE_API_SECRET")  or "yy1wd2wn8r0wx4mus00vxllgss03nuqx"
ACCESS_TOKEN= os.getenv("KITE_ACCESS_TOKEN")or "bx30fqj4sPchTK6KuSuC9JU6L4pFSnmR"  # refresh daily
INSTRUMENTS_CSV = os.getenv("KITE_INSTR_CSV") or "instruments.csv"     # download once

# Symbol list in TradingView style; we’ll split on ":" and use tradingsymbol + exchange mapping
SYMBOLS = ["NSE:TCS","NSE:RELIANCE"]  # add more

START_DATE = "2024-01-01 09:15:00"
END_DATE   = "2025-09-30 15:30:00"
INTERVAL   = "60minute"   # hourly [web:140][web:145]

# Strategy params
RISK_PCT_EQUITY = 1.0
SLIPPAGE_POINTS = 0.10
MAX_SHARES_CAP  = 100000
INITIAL_EQUITY  = 1_000_000.0

SL_TYPE   = "OppositeRange"  # OppositeRange | FixedPoints | FixedPct | ATR
SL_VALUE  = 0.0              # pts / % / ATR multiple (ignored for OppositeRange)
TP_TYPE   = "RR"             # RR | FixedPoints | FixedPct | ATR
TP_VALUE  = 2.0              # multiple or pts/%/ATR
ATR_LEN   = 14

ALLOW_SECOND_CHANCE = True
SESSION_EOD = dt.time(15,30)
IST = pytz.timezone("Asia/Kolkata")

OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

# ========== Utilities ==========
def kite_client():
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite

def load_instruments():
    # https://api.kite.trade/instruments (CSV)
    df = pd.read_csv(INSTRUMENTS_CSV)
    # Keep NSE equities
    df = df[(df["exchange"]=="NSE") & (df["segment"].str.contains("NSE"))]
    return df

def token_for(kite_instr_df: pd.DataFrame, tv_symbol: str) -> int:
    # tv_symbol: "NSE:TCS" => exchange="NSE", tradingsymbol="TCS"
    exc, tsym = tv_symbol.split(":")
    row = kite_instr_df[(kite_instr_df["exchange"]==exc) & (kite_instr_df["tradingsymbol"]==tsym)]
    if row.empty:
        raise ValueError(f"No instrument token for {tv_symbol}")
    return int(row.iloc[0]["instrument_token"])

def chunked_historical(kite: KiteConnect, token: int, start: dt.datetime, end: dt.datetime, interval: str):
    # Kite historical has practical range limits; chunk by 60 days for safety [web:141]
    out = []
    cur = start
    while cur < end:
        to = min(cur + dt.timedelta(days=60), end)
        data = kite.historical_data(token, cur, to, interval, continuous=False, oi=False)  # [web:140]
        if data:
            out.extend(data)
        cur = to
        _time.sleep(0.2)  # be kind to rate limits
    # Convert to DataFrame
    if not out:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    x = pd.DataFrame(out)
    # Kite returns 'date' tz-aware IST strings ISO-like
    x["date"] = pd.to_datetime(x["date"])
    x = x.set_index("date").sort_index()
    return x[["open","high","low","close","volume"]]

def compute_atr(df: pd.DataFrame, n:int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(),
                    (df["high"]-pc).abs(),
                    (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def ist_time(ts: pd.Timestamp) -> dt.time:
    return ts.tz_convert(IST).time()

# ========== Strategy (Hourly ORB rolling) ==========
def backtest_symbol_hourly(df: pd.DataFrame, symbol: str):
    # df indexed by tz-aware timestamps (IST from Kite)
    df = df.copy().sort_index()
    df["atr"] = compute_atr(df, ATR_LEN)

    equity = INITIAL_EQUITY
    in_pos = False
    qty = 0
    entry = np.nan
    last_ref_high = np.nan
    last_ref_low = np.nan
    reentry_used = False
    trades = []

    # We use prior hour bar as reference; on each bar close, check breakout vs previous bar high
    prev = None
    for ts, row in df.iterrows():
        # EOD square-off
        if in_pos and ist_time(ts) >= SESSION_EOD:
            exit_px = row["close"]
            pnl = (exit_px - entry) * qty
            equity += pnl
            trades.append(dict(ts=ts, symbol=symbol, side="EOD_CLOSE", entry=entry, exit=exit_px, qty=qty, pnl=pnl, equity=equity))
            in_pos = False; qty=0; entry=np.nan

        # Need at least one completed prior bar to define reference
        if prev is None:
            prev = row
            continue

        ref_high = float(prev["high"])
        ref_low  = float(prev["low"])
        last_ref_high = ref_high
        last_ref_low  = ref_low

        price = float(row["close"])
        atr   = float(row["atr"]) if not math.isnan(row["atr"]) else None

        # If in position, manage exits
        if in_pos:
            # Stop
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

            exit_reason = None
            exit_px = None
            if price <= stop:
                exit_reason = "SL"
                exit_px = stop
            elif price >= target:
                exit_reason = "TP"
                exit_px = target

            if exit_reason:
                pnl = (exit_px - entry) * qty
                equity += pnl
                trades.append(dict(ts=ts, symbol=symbol, side=exit_reason, entry=entry, exit=exit_px, qty=qty, pnl=pnl, equity=equity))
                in_pos = False; qty=0; entry=np.nan
                if exit_reason=="SL":
                    reentry_used = False  # eligible for one re-entry on later breakout
                else:
                    reentry_used = True   # no re-entry after TP

        # If flat, check breakout over previous hour high
        if not in_pos:
            breakout = price > ref_high
            can_enter = breakout and (not reentry_used or ALLOW_SECOND_CHANCE)
            if can_enter:
                # Compute stop for sizing as if OppositeRange unless different SL_TYPE
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
                    entry = ref_high  # realistic: stop limit at breakout; you may prefer price
                    trades.append(dict(ts=ts, symbol=symbol, side="BUY", entry=entry, exit=np.nan, qty=qty, pnl=0.0, equity=equity))
                    # After first entry for this reference, block another immediate same-bar entry
                    reentry_used = True if not ALLOW_SECOND_CHANCE else reentry_used

        prev = row

    return pd.DataFrame(trades)

def main():
    kite = kite_client()
    instr = load_instruments()
    start = pd.Timestamp(START_DATE).tz_localize(IST)
    end   = pd.Timestamp(END_DATE).tz_localize(IST)

    all_trades = []
    for sym in SYMBOLS:
        token = token_for(instr, sym)
        df = chunked_historical(kite, token, start.to_pydatetime(), end.to_pydatetime(), INTERVAL)
        if df.empty:
            print(f"No data for {sym}")
            continue
        # Kite dates are already IST tz-aware; ensure tz
        if df.index.tz is None:
            df.index = df.index.tz_localize(IST)
        trades = backtest_symbol_hourly(df, sym)
        trades["symbol_tv"] = sym
        all_trades.append(trades)

    if not all_trades:
        print("No trades.")
        return

    trades = pd.concat(all_trades).sort_values("ts")
    trades.to_csv(os.path.join(OUT_DIR, "trades_hourly_orb.csv"), index=False)

    # Build equity curve from FIRST symbol’s starting equity, aggregating pnl chronologically
    equity = INITIAL_EQUITY
    curve = []
    for _, r in trades.iterrows():
        if pd.notna(r["exit"]):
            equity = r["equity"]
        curve.append(dict(ts=r["ts"], equity=r["equity"]))
    pd.DataFrame(curve).drop_duplicates("ts").to_csv(os.path.join(OUT_DIR, "equity_curve.csv"), index=False)

    print(f"Saved trades to out/trades_hourly_orb.csv and equity to out/equity_curve.csv")

if __name__ == "__main__":
    main()
