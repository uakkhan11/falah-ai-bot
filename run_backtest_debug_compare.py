import os
import pandas as pd
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta

# ===== CONFIG =====
HISTORICAL_DIR = "/root/falah-ai-bot/historical_data/"
MODEL_PATH = "/root/falah-ai-bot/model.pkl"
START_CAPITAL = 1_000_000
RISK_PER_TRADE = 0.02   # 2% risk
MAX_QTY = 5000
SL_PCT = 0.05           # 5% Stop Loss
TP_PCT = 0.20           # 20% Take Profit
TRAIL_MULT = 3.0        # ATR trailing multiplier
MAX_VOL_CHANGE = 0.5    # Avoid volatile entries

FEATURES = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"]

model = joblib.load(MODEL_PATH)

indicator_pass_count = {"RSI": 0, "EMA": 0, "Supertrend": 0, "AI_Score": 0}
skip_reason_count = {}
exit_reason_count = {}

def track_skip(reason):
    skip_reason_count[reason] = skip_reason_count.get(reason, 0) + 1

def track_exit(reason):
    exit_reason_count[reason] = exit_reason_count.get(reason, 0) + 1

def calculate_features(df):
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

    df["EMA10"] = ta.ema(df["close"], length=10)
    df["EMA21"] = ta.ema(df["close"], length=21)
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["VolumeChange"] = df["volume"].pct_change()

    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["Supertrend"] = st["SUPERT_10_3.0"]

    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df

def apply_ai_score(df):
    X = df[FEATURES].copy()
    X = X.replace([float("inf"), float("-inf")], pd.NA).dropna()
    df = df.loc[X.index].copy()
    X = X.astype("float32")
    df["ai_score"] = model.predict_proba(X)[:, 1]
    return df

def run_backtest():
    capital = START_CAPITAL
    trades = []
    cutoff_date = datetime.now() - timedelta(days=730)  # Last 2 years

    for file in os.listdir(HISTORICAL_DIR):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(HISTORICAL_DIR, file))
        df.columns = [c.lower() for c in df.columns]

        if "date" not in df.columns:
            track_skip("No date col")
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] >= cutoff_date]
        if len(df) < 50:
            track_skip("Insufficient data")
            continue

        df = calculate_features(df)
        if len(df) < 50:
            track_skip("Insufficient data")
            continue

        df = apply_ai_score(df)

        for i in range(len(df) - 1):
            row = df.iloc[i]

            rsi_pass = 30 < row["RSI"] < 70
            ema_pass = row["EMA10"] > row["EMA21"]
            st_pass = row["close"] > row["Supertrend"]
            ai_pass = row["ai_score"] > 0.20

            if row["VolumeChange"] > MAX_VOL_CHANGE:
                track_skip("Volatile entry")
                continue

            if not rsi_pass: track_skip("RSI fail")
            if not ema_pass: track_skip("EMA fail")
            if not st_pass: track_skip("Supertrend fail")
            if not ai_pass: track_skip("AI score fail")

            if not (rsi_pass and ema_pass and st_pass and ai_pass):
                continue

            indicator_pass_count["RSI"] += 1
            indicator_pass_count["EMA"] += 1
            indicator_pass_count["Supertrend"] += 1
            indicator_pass_count["AI_Score"] += 1

            entry_price = row["close"]
            atr = row["ATR"]
            if atr <= 0:
                continue

            qty = int(min((capital * RISK_PER_TRADE) / atr, MAX_QTY))
            if qty <= 0:
                track_skip("Zero qty")
                continue

            sl_price = entry_price * (1 - SL_PCT)
            tp_price = entry_price * (1 + TP_PCT)
            trail_sl = sl_price
            breakeven = False

            for j in range(i + 1, len(df)):
                exit_row = df.iloc[j]
                ltp = exit_row["close"]

                # Breakeven trigger
                if not breakeven and ltp > entry_price:
                    breakeven = True

                # Trailing SL only after breakeven
                if breakeven:
                    atr_val = exit_row["ATR"]
                    trail_sl = max(trail_sl, ltp - TRAIL_MULT * atr_val)

                reason = None
                if ltp <= sl_price:
                    reason = f"Fixed SL breach (-{SL_PCT*100:.0f}%)"
                elif breakeven and ltp <= trail_sl:
                    reason = "Trailing SL breached"
                elif ltp >= tp_price:
                    reason = f"Profit >={TP_PCT*100:.0f}% hit"

                if reason:
                    pnl = (ltp - entry_price) * qty
                    capital += pnl
                    trades.append({
                        "symbol": symbol,
                        "entry_date": df["date"].iloc[i],
                        "exit_date": df["date"].iloc[j],
                        "entry_price": entry_price,
                        "exit_price": ltp,
                        "qty": qty,
                        "pnl": pnl,
                        "reason": reason
                    })
                    track_exit(reason)
                    break

    total_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    pnl_total = sum(t["pnl"] for t in trades)

    print(f"\nPeriod: Last 2 years")
    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {wins} ({wins / total_trades * 100:.2f}%)")
    print(f"Total PnL: ₹{pnl_total:,.2f}")
    print(f"Final Capital: ₹{capital:,.2f}")

    print("\nIndicator Pass Counts:")
    for k, v in indicator_pass_count.items():
        print(f"{k}: {v}")

    print("\nSkip Reasons:")
    for k, v in skip_reason_count.items():
        print(f"{k}: {v}")

    print("\nExit Reason Counts:")
    for k, v in exit_reason_count.items():
        print(f"{k}: {v}")

    pd.DataFrame(trades).to_csv("backtest_trades.csv", index=False)
    print("\n✅ backtest_trades.csv saved.")

if __name__ == "__main__":
    run_backtest()
