# backtester.py

import os
import pandas as pd
from stock_analysis import analyze_stock
from credentials import get_kite, validate_kite

# Load list of symbols to backtest
symbols = ["INFY", "TCS", "HDFCBANK"]  # You can customize this

kite = get_kite()
if not validate_kite(kite):
    print("❌ Invalid Kite token. Exiting.")
    exit()

results = []

for symbol in symbols:
    try:
        print(f"🔍 Backtesting {symbol}...")
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
        df["Close_shift"] = df["close"].shift(-1)  # Next day's close for exit

        trades = []

        for i in range(len(df) - 1):
            slice_df = df.iloc[: i + 1].reset_index()
            last_close = slice_df["close"].iloc[-1]
            trailing_sl = last_close - slice_df["close"].std() * 1.5

            # Run AI exit score
            from ai_engine import calculate_ai_exit_score
            ai_score, reasons = calculate_ai_exit_score(
                stock_data=slice_df,
                trailing_sl=trailing_sl,
                current_price=last_close
            )

            if ai_score >= 70:
                exit_price = df["Close_shift"].iloc[i]
                profit = exit_price - last_close
                trades.append({
                    "date": slice_df["date"].iloc[-1],
                    "entry": last_close,
                    "exit": exit_price,
                    "profit": profit,
                    "ai_score": ai_score,
                    "reasons": "|".join(reasons)
                })

        if trades:
            trade_df = pd.DataFrame(trades)
            win_rate = (trade_df["profit"] > 0).mean() * 100
            avg_profit = trade_df["profit"].mean()
            print(f"✅ {symbol}: {len(trades)} trades, Win Rate {win_rate:.1f}%, Avg Profit {avg_profit:.2f}")
            trade_df.to_csv(f"backtest_{symbol}.csv", index=False)
            results.append({
                "symbol": symbol,
                "trades": len(trade_df),
                "win_rate": win_rate,
                "avg_profit": avg_profit
            })
        else:
            print(f"⚠️ No exits triggered for {symbol}.")
    except Exception as e:
        print(f"❌ Error backtesting {symbol}: {e}")

summary_df = pd.DataFrame(results)
summary_df.to_csv("backtest_summary.csv", index=False)
print("🎯 Backtesting complete. Summary saved to backtest_summary.csv.")
