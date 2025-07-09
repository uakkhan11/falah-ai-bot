# backtester_full.py

import os
import pandas as pd
from datetime import datetime
from credentials import get_kite, validate_kite
from ai_engine import calculate_ai_exit_score
from stock_analysis import analyze_stock

# 🟢 Parameters
symbols = ["INFY", "TCS", "HDFCBANK"]
initial_capital = 1000000  # ₹10 lakh starting capital
risk_per_trade_pct = 2  # 2% risk per trade
rr_ratio = 3  # 1:3 R:R
atr_multiplier_sl = 1.5

# 🟢 Kite
kite = get_kite()
if not validate_kite(kite):
    print("❌ Invalid Kite token.")
    exit()

results = []
capital = initial_capital
equity_curve = []

# 🟢 Backtest each symbol
for symbol in symbols:
    print(f"\n🔍 Backtesting {symbol}...")

    # Load historical data
    hist = kite.historical_data(
        instrument_token=kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"],
        from_date="2023-01-01",
        to_date="2023-12-31",
        interval="day"
    )
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Compute indicators
    df["SMA20"] = df["close"].rolling(20).mean()
    df["EMA10"] = df["close"].ewm(span=10).mean()
    df["EMA21"] = df["close"].ewm(span=21).mean()
    df["RSI"] = pd.Series(dtype=float)

    import ta
    df["RSI"] = ta.momentum.rsi(df["close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    trades = []

    for i in range(21, len(df) - 1):
        # Entry conditions (example)
        if (
            df["close"].iloc[i] > df["SMA20"].iloc[i]
            and df["EMA10"].iloc[i] > df["EMA21"].iloc[i]
            and df["RSI"].iloc[i] > 55
        ):
            entry_date = df.index[i]
            entry_price = df["close"].iloc[i]
            atr = df["ATR"].iloc[i]
            sl = entry_price - atr * atr_multiplier_sl
            target = entry_price + (entry_price - sl) * rr_ratio

            # Risk-based sizing
            per_share_risk = entry_price - sl
            position_risk = capital * (risk_per_trade_pct / 100)
            qty = int(position_risk / per_share_risk)
            if qty <= 0:
                continue

            # Simulate holding until exit
            exit_price = None
            exit_reason = ""
            for j in range(i + 1, len(df)):
                date_j = df.index[j]
                close_j = df["close"].iloc[j]
                atr_j = df["ATR"].iloc[j]
                trailing_sl = max(sl, close_j - atr_j * atr_multiplier_sl)

                # AI exit
                slice_df = df.iloc[:j + 1].reset_index()
                ai_score, reasons = calculate_ai_exit_score(
                    stock_data=slice_df,
                    trailing_sl=trailing_sl,
                    current_price=close_j
                )
                if ai_score >= 70:
                    exit_price = close_j
                    exit_reason = "AI Exit"
                    break

                # Target hit
                if close_j >= target:
                    exit_price = target
                    exit_reason = "Target Hit"
                    break

                # Trailing SL hit
                if close_j <= trailing_sl:
                    exit_price = trailing_sl
                    exit_reason = "Stop Loss"
                    break

            if not exit_price:
                # Exit at final candle if never exited
                exit_price = df["close"].iloc[-1]
                exit_reason = "End of Data"

            profit = (exit_price - entry_price) * qty
            capital += profit
            equity_curve.append(capital)

            print(f"✅ {symbol} Trade: {entry_date.date()} to {date_j.date()} | {exit_reason} | P&L: ₹{profit:.2f}")
            trades.append({
                "Entry Date": entry_date,
                "Exit Date": date_j,
                "Entry": entry_price,
                "Exit": exit_price,
                "Reason": exit_reason,
                "Qty": qty,
                "Profit": profit
            })

    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades.to_csv(f"backtest_{symbol}.csv", index=False)
        win_rate = (df_trades["Profit"] > 0).mean() * 100
        avg_profit = df_trades["Profit"].mean()
        results.append({
            "Symbol": symbol,
            "Trades": len(df_trades),
            "Win Rate": win_rate,
            "Avg Profit": avg_profit
        })
    else:
        print(f"⚠️ No trades triggered for {symbol}.")

# 🟢 Save equity curve
equity_df = pd.DataFrame({"Capital": equity_curve})
equity_df.to_csv("backtest_equity_curve.csv", index=False)

# 🟢 Summary
summary_df = pd.DataFrame(results)
summary_df.to_csv("backtest_summary.csv", index=False)

print("\n🎯 Backtest Complete. Results saved:")
print(summary_df)
