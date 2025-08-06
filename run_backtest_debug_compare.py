import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
import joblib
import pandas_ta as ta

# === CONFIG ===
HISTORICAL_PATH = "/root/falah-ai-bot/historical_data"
MODEL_PATH = "/root/falah-ai-bot/model.pkl"
PERIOD_YEARS = 2
ATR_MULTIPLIER = 1.5
STAGE1_TARGET_PCT = 0.05   # +5% partial exit
FINAL_TARGET_PCT = 0.08    # +8% full target
PARTIAL_EXIT_RATIO = 0.5   # 50% exit at stage 1
CSV_TRADE_LOG = "backtest_trades.csv"

# Load AI model
model = joblib.load(MODEL_PATH)

# Stats tracking
primary_trades = 0
primary_wins = 0
bb_trades = 0
bb_wins = 0
indicator_pass_counts = {"RSI": 0, "EMA": 0, "Supertrend": 0, "AI_Score": 0}
skip_reasons = {"AI score fail": 0, "EMA fail": 0, "RSI fail": 0, "Supertrend fail": 0, "Insufficient data": 0}
exit_reasons = {
    "Initial SL hit": 0,
    "Stage1 Hit": 0,
    "Final Target": 0,
    "Trailing SL pre-Stage1": 0,
    "Trailing SL after Stage1": 0,
    "BB SL": 0,
    "BB Target": 0
}

def calculate_indicators(df):
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["ema10"] = ta.ema(df["close"], length=10)
    df["ema21"] = ta.ema(df["close"], length=21)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["supertrend"] = st["SUPERTd_10_3.0"]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["volumechange"] = df["volume"].pct_change().fillna(0)
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_lower"] = bb["BBL_20_2.0"]
    df["bb_upper"] = bb["BBU_20_2.0"]
    return df

def run_backtest():
    global primary_trades, primary_wins, bb_trades, bb_wins

    cutoff_date = datetime.now(pytz.timezone("Asia/Kolkata")) - timedelta(days=PERIOD_YEARS * 365)

    for file in os.listdir(HISTORICAL_PATH):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(HISTORICAL_PATH, file))
        except Exception:
            skip_reasons["Insufficient data"] += 1
            continue

        if "date" not in df.columns:
            skip_reasons["Insufficient data"] += 1
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("Asia/Kolkata", nonexistent="NaT")
        else:
            df["date"] = df["date"].dt.tz_convert("Asia/Kolkata")

        df = df[df["date"] >= cutoff_date]

        if len(df) < 50:
            skip_reasons["Insufficient data"] += 1
            continue

        df = calculate_indicators(df)
        df.dropna(inplace=True)

        for i in range(len(df)):
            row = df.iloc[i]

            # === Primary Strategy ===
            if 35 <= row["rsi"] <= 70 and row["ema10"] > row["ema21"] and row["supertrend"] == 1:
                indicator_pass_counts["RSI"] += 1
                indicator_pass_counts["EMA"] += 1
                indicator_pass_counts["Supertrend"] += 1

                features_df = pd.DataFrame([[
                    row["rsi"], row["atr"], row["adx"] if "adx" in df.columns else 0,
                    row["ema10"], row["ema21"], row["volumechange"]
                ]], columns=["rsi", "atr", "adx", "ema10", "ema21", "volumechange"])

                try:
                    ai_score = model.predict_proba(features_df)[0][1]
                except:
                    skip_reasons["AI score fail"] += 1
                    continue

                if ai_score < 0.25:
                    skip_reasons["AI score fail"] += 1
                    continue
                indicator_pass_counts["AI_Score"] += 1

                entry_price = row["close"]
                atr_value = row["atr"]
                stop_loss_price = entry_price - ATR_MULTIPLIER * atr_value
                stage1_target_price = entry_price * (1 + STAGE1_TARGET_PCT)
                final_target_price = entry_price * (1 + FINAL_TARGET_PCT)
                trailing_sl = stop_loss_price

                stage1_hit = False
                primary_trades += 1

                for j in range(i + 1, len(df)):
                    future_price = df.iloc[j]["close"]

                    # Stage 1 partial profit
                    if not stage1_hit and future_price >= stage1_target_price:
                        stage1_hit = True
                        exit_reasons["Stage1 Hit"] += 1
                        trailing_sl = max(trailing_sl, future_price - ATR_MULTIPLIER * atr_value)
                        continue

                    # Final target
                    if future_price >= final_target_price:
                        primary_wins += 1
                        exit_reasons["Final Target"] += 1
                        break

                    # Trailing SL
                    trailing_sl = max(trailing_sl, future_price - ATR_MULTIPLIER * atr_value)
                    if future_price <= trailing_sl:
                        if stage1_hit:
                            exit_reasons["Trailing SL after Stage1"] += 1
                        else:
                            exit_reasons["Trailing SL pre-Stage1"] += 1
                        break

                    # Initial SL hit
                    if future_price <= stop_loss_price:
                        exit_reasons["Initial SL hit"] += 1
                        break

            # === Fallback BB Strategy ===
            elif row["close"] <= row["bb_lower"]:
                bb_trades += 1
                entry_price = row["close"]
                stop_loss_price = row["bb_lower"] * 0.98
                target_price = row["bb_upper"]

                for j in range(i + 1, len(df)):
                    future_price = df.iloc[j]["close"]

                    if future_price <= stop_loss_price:
                        exit_reasons["BB SL"] += 1
                        break

                    if future_price >= target_price:
                        bb_wins += 1
                        exit_reasons["BB Target"] += 1
                        break

    # === Summary ===
    total_trades = primary_trades + bb_trades
    total_wins = primary_wins + bb_wins
    primary_win_pct = (primary_wins / primary_trades * 100) if primary_trades > 0 else 0
    bb_win_pct = (bb_wins / bb_trades * 100) if bb_trades > 0 else 0
    overall_win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print("\n=== BACKTEST SUMMARY ===")
    print(f"Period: Last {PERIOD_YEARS} years")
    print(f"Total Trades: {total_trades}")
    print(f"Primary Trades: {primary_trades} | Wins: {primary_wins} ({primary_win_pct:.2f}%)")
    print(f"BB Trades: {bb_trades} | Wins: {bb_wins} ({bb_win_pct:.2f}%)")
    print(f"Overall Win %: {overall_win_pct:.2f}%\n")

    print("Indicator Pass Counts:")
    for k, v in indicator_pass_counts.items():
        print(f"  {k}: {v}")

    print("\nSkip Reasons:")
    for k, v in skip_reasons.items():
        print(f"  {k}: {v}")

    print("\nExit Reason Counts:")
    for k, v in exit_reasons.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    run_backtest()
