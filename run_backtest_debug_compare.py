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
TARGET_PROFIT_PCT = 0.08
STOP_LOSS_PCT = 0.03
TRAILING_SL_MULTIPLIER = 1.5
POSITION_SIZE = 10000  # ₹ per trade
BROKERAGE_PCT = 0.001  # 0.1%
SLIPPAGE_PCT = 0.0005  # 0.05%

# Load AI model
model = joblib.load(MODEL_PATH)

# Summary stats
overall_results = {
    "primary_trades": 0, "primary_wins": 0,
    "bb_trades": 0, "bb_wins": 0,
    "capital": 100000, "pnl": 0
}
indicator_pass_counts = {"RSI": 0, "EMA": 0, "Supertrend": 0, "AI_Score": 0}
skip_reasons = {"AI score fail": 0, "EMA fail": 0, "RSI fail": 0, "Supertrend fail": 0, "Insufficient data": 0}
exit_reasons = {
    "Fixed SL": 0, "Target": 0, "Trailing SL": 0,
    "BB SL": 0, "BB Target": 0
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
    df["BB_lower"] = bb["BBL_20_2.0"]
    df["BB_upper"] = bb["BBU_20_2.0"]
    return df

def run_backtest():
    cutoff_date = datetime.now(pytz.timezone("Asia/Kolkata")) - timedelta(days=PERIOD_YEARS * 365)
    start_capital = overall_results["capital"]

    for file in os.listdir(HISTORICAL_PATH):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(HISTORICAL_PATH, file))
        except:
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

        symbol_stats = {
            "primary_trades": 0, "primary_wins": 0,
            "bb_trades": 0, "bb_wins": 0,
            "skip": skip_reasons.copy(),
            "exit": exit_reasons.copy(),
            "pnl": 0
        }

        open_trade = False

        for i in range(len(df)):
            if open_trade:
                continue  # only one active trade at a time per symbol

            row = df.iloc[i]

            # === Primary Strategy ===
            if 35 <= row["RSI"] <= 70 and row["EMA10"] > row["EMA21"] and row["supertrend"] == 1:
                indicator_pass_counts["RSI"] += 1
                indicator_pass_counts["EMA"] += 1
                indicator_pass_counts["Supertrend"] += 1

                features_df = pd.DataFrame([[row["rsi"], row["atr"], row["adx"] if "adx" in df.columns else 0,
                             row["ema10"], row["ema21"], row["volumechange"]]],
                           columns=["rsi", "atr", "adx", "ema10", "ema21", "volumechange"])

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
                qty = POSITION_SIZE / entry_price
                atr_value = row["ATR"]
                stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                target_price = entry_price * (1 + TARGET_PROFIT_PCT)
                trailing_sl = entry_price - TRAILING_SL_MULTIPLIER * atr_value

                for j in range(i + 1, len(df)):
                    future_price = df.iloc[j]["close"]

                    if future_price <= stop_loss_price:
                        pnl = (future_price - entry_price) * qty
                        pnl -= POSITION_SIZE * (BROKERAGE_PCT + SLIPPAGE_PCT) * 2
                        overall_results["pnl"] += pnl
                        symbol_stats["pnl"] += pnl
                        symbol_stats["primary_trades"] += 1
                        exit_reasons["Fixed SL"] += 1
                        break

                    if future_price >= target_price:
                        pnl = (future_price - entry_price) * qty
                        pnl -= POSITION_SIZE * (BROKERAGE_PCT + SLIPPAGE_PCT) * 2
                        overall_results["pnl"] += pnl
                        symbol_stats["pnl"] += pnl
                        symbol_stats["primary_trades"] += 1
                        symbol_stats["primary_wins"] += 1
                        exit_reasons["Target"] += 1
                        break

                    trailing_sl = max(trailing_sl, future_price - TRAILING_SL_MULTIPLIER * atr_value)
                    if future_price <= trailing_sl:
                        pnl = (future_price - entry_price) * qty
                        pnl -= POSITION_SIZE * (BROKERAGE_PCT + SLIPPAGE_PCT) * 2
                        overall_results["pnl"] += pnl
                        symbol_stats["pnl"] += pnl
                        symbol_stats["primary_trades"] += 1
                        exit_reasons["Trailing SL"] += 1
                        break

                open_trade = True

            # === Fallback BB Strategy ===
            elif row["close"] <= row["BB_lower"]:
                qty = POSITION_SIZE / row["close"]
                stop_loss_price = row["BB_lower"] * 0.98
                target_price = row["BB_upper"]

                for j in range(i + 1, len(df)):
                    future_price = df.iloc[j]["close"]

                    if future_price <= stop_loss_price:
                        pnl = (future_price - row["close"]) * qty
                        pnl -= POSITION_SIZE * (BROKERAGE_PCT + SLIPPAGE_PCT) * 2
                        overall_results["pnl"] += pnl
                        symbol_stats["pnl"] += pnl
                        symbol_stats["bb_trades"] += 1
                        exit_reasons["BB SL"] += 1
                        break

                    if future_price >= target_price:
                        pnl = (future_price - row["close"]) * qty
                        pnl -= POSITION_SIZE * (BROKERAGE_PCT + SLIPPAGE_PCT) * 2
                        overall_results["pnl"] += pnl
                        symbol_stats["pnl"] += pnl
                        symbol_stats["bb_trades"] += 1
                        symbol_stats["bb_wins"] += 1
                        exit_reasons["BB Target"] += 1
                        break

                open_trade = True

        # Micro-summary for this symbol
        print(f"\n📌 {symbol} Summary:")
        print(f" Primary Trades: {symbol_stats['primary_trades']} | Wins: {symbol_stats['primary_wins']}")
        print(f" BB Trades: {symbol_stats['bb_trades']} | Wins: {symbol_stats['bb_wins']}")
        print(f" PnL: ₹{symbol_stats['pnl']:.2f}")

    # Overall summary
    end_capital = start_capital + overall_results["pnl"]
    print("\n====== OVERALL SUMMARY ======")
    print(f"Start Capital: ₹{start_capital:,.2f}")
    print(f"End Capital: ₹{end_capital:,.2f}")
    print(f"Net Profit: ₹{overall_results['pnl']:,.2f}")
    print(f"Total Trades: {overall_results['primary_trades'] + overall_results['bb_trades']}")

if __name__ == "__main__":
    run_backtest()
