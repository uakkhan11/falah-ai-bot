import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import joblib
import pandas_ta as ta

# === CONFIG ===
HISTORICAL_PATH = "/root/falah-ai-bot/historical_data"
MODEL_PATH = "/root/falah-ai-bot/model.pkl"
PERIOD_YEARS = 2
TARGET_PROFIT_PCT = 0.08   # 8% target
STOP_LOSS_PCT = 0.03       # 3% fixed SL
TRAILING_SL_MULTIPLIER = 1.5
START_CAPITAL = 100000
TRADE_RISK = 0.02          # 2% of capital risk per trade
CSV_TRADE_LOG = "backtest_trades.csv"
CSV_SUMMARY = "backtest_summary.csv"

# Load AI model
model = joblib.load(MODEL_PATH)

# Summary stats
primary_trades = 0
primary_wins = 0
bb_trades = 0
bb_wins = 0
total_pnl = 0
capital = START_CAPITAL
equity_curve = [capital]
indicator_pass_counts = {"RSI": 0, "EMA": 0, "Supertrend": 0, "AI_Score": 0}
skip_reasons = {"AI score fail": 0, "EMA fail": 0, "RSI fail": 0, "Supertrend fail": 0, "Insufficient data": 0}
exit_reasons = {
    "Fixed SL breach (-3%)": 0,
    "Profit >=8% hit": 0,
    "Trailing SL breached": 0,
    "BB SL": 0,
    "BB Target": 0
}
trade_log = []

def calculate_indicators(df):
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["EMA10"] = ta.ema(df["close"], length=10)
    df["EMA21"] = ta.ema(df["close"], length=21)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["supertrend"] = st["SUPERTd_10_3.0"]
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["VolumeChange"] = df["volume"].pct_change().fillna(0)
    macd = ta.macd(df["close"])
    df["MACD_Hist"] = macd["MACDh_12_26_9"]

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20, std=2)
    df["BB_lower"] = bb["BBL_20_2.0"]
    df["BB_upper"] = bb["BBU_20_2.0"]

    return df

def run_backtest():
    global primary_trades, primary_wins, bb_trades, bb_wins, total_pnl, capital

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

        # Fix date parsing
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

            # Primary strategy
            if 35 <= row["RSI"] <= 70 and row["EMA10"] > row["EMA21"] and row["supertrend"] == 1:
                indicator_pass_counts["RSI"] += 1
                indicator_pass_counts["EMA"] += 1
                indicator_pass_counts["Supertrend"] += 1

                features_df = pd.DataFrame([[row["RSI"], row["ATR"], row.get("ADX", 0),
                                             row["EMA10"], row["EMA21"], row["VolumeChange"]]],
                                           columns=["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"])

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
                atr_value = row["ATR"]
                qty = int((capital * TRADE_RISK) / (STOP_LOSS_PCT * entry_price))
                stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                target_price = entry_price * (1 + TARGET_PROFIT_PCT)
                trailing_sl = entry_price - TRAILING_SL_MULTIPLIER * atr_value

                for j in range(i + 1, len(df)):
                    future_price = df.iloc[j]["close"]

                    if future_price <= stop_loss_price:
                        pnl = (future_price - entry_price) * qty
                        capital += pnl
                        equity_curve.append(capital)
                        trade_log.append([symbol, df.iloc[i]["date"], entry_price, future_price, qty, pnl, "SL"])
                        primary_trades += 1
                        exit_reasons["Fixed SL breach (-3%)"] += 1
                        break

                    if future_price >= target_price:
                        pnl = (future_price - entry_price) * qty
                        capital += pnl
                        equity_curve.append(capital)
                        trade_log.append([symbol, df.iloc[i]["date"], entry_price, future_price, qty, pnl, "Target"])
                        primary_trades += 1
                        primary_wins += 1
                        exit_reasons["Profit >=8% hit"] += 1
                        break

                    trailing_sl = max(trailing_sl, future_price - TRAILING_SL_MULTIPLIER * atr_value)
                    if future_price <= trailing_sl:
                        pnl = (future_price - entry_price) * qty
                        capital += pnl
                        equity_curve.append(capital)
                        trade_log.append([symbol, df.iloc[i]["date"], entry_price, future_price, qty, pnl, "Trailing SL"])
                        primary_trades += 1
                        exit_reasons["Trailing SL breached"] += 1
                        break

            # Fallback BB strategy
            elif row["close"] <= row["BB_lower"]:
                qty = int((capital * TRADE_RISK) / (STOP_LOSS_PCT * row["close"]))
                entry_price = row["close"]
                stop_loss_price = row["BB_lower"] * 0.98
                target_price = row["BB_upper"]

                for j in range(i + 1, len(df)):
                    future_price = df.iloc[j]["close"]

                    if future_price <= stop_loss_price:
                        pnl = (future_price - entry_price) * qty
                        capital += pnl
                        equity_curve.append(capital)
                        trade_log.append([symbol, df.iloc[i]["date"], entry_price, future_price, qty, pnl, "BB SL"])
                        bb_trades += 1
                        exit_reasons["BB SL"] += 1
                        break

                    if future_price >= target_price:
                        pnl = (future_price - entry_price) * qty
                        capital += pnl
                        equity_curve.append(capital)
                        trade_log.append([symbol, df.iloc[i]["date"], entry_price, future_price, qty, pnl, "BB Target"])
                        bb_trades += 1
                        bb_wins += 1
                        exit_reasons["BB Target"] += 1
                        break

    # === Performance Metrics ===
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 else 0
    max_drawdown = ((np.max(equity_curve) - np.min(equity_curve)) / np.max(equity_curve)) * 100
    gross_profit = sum([t[5] for t in trade_log if t[5] > 0])
    gross_loss = abs(sum([t[5] for t in trade_log if t[5] < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    net_profit_pct = ((capital - START_CAPITAL) / START_CAPITAL) * 100

    # === Print Summary ===
    print(f"Final Capital: ₹{capital:,.2f}")
    print(f"Net Profit %: {net_profit_pct:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Total Trades: {len(trade_log)}")
    print(f"Primary Trades: {primary_trades} | Wins: {primary_wins}")
    print(f"BB Trades: {bb_trades} | Wins: {bb_wins}")

    # === Save Logs ===
    pd.DataFrame(trade_log, columns=["Symbol", "Entry Date", "Entry Price", "Exit Price", "Qty", "PnL", "Exit Reason"]).to_csv(CSV_TRADE_LOG, index=False)
    pd.DataFrame([{
        "Final Capital": capital,
        "Net Profit %": net_profit_pct,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown %": max_drawdown,
        "Profit Factor": profit_factor,
        "Total Trades": len(trade_log),
        "Primary Trades": primary_trades,
        "Primary Wins": primary_wins,
        "BB Trades": bb_trades,
        "BB Wins": bb_wins
    }]).to_csv(CSV_SUMMARY, index=False)

if __name__ == "__main__":
    run_backtest()
