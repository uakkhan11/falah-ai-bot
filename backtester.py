# backtester.py

import os
import pandas as pd
from datetime import datetime

class Backtester:
    def __init__(self, data_dir, symbols, initial_capital=100000, slippage_pct=0.002, commission_per_trade=20, risk_per_trade_pct=2):
        """
        data_dir: Folder with historical CSVs per symbol
        symbols: List of symbols to backtest
        """
        self.data_dir = data_dir
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_per_trade = commission_per_trade
        self.risk_per_trade_pct = risk_per_trade_pct
        self.results = []

    def load_data(self, symbol):
        filepath = os.path.join(self.data_dir, f"{symbol}.csv")
        if not os.path.exists(filepath):
            print(f"⚠️ Missing data for {symbol}")
            return None
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    def run(self):
        equity = self.initial_capital

        for symbol in self.symbols:
            df = self.load_data(symbol)
            if df is None or len(df) < 30:
                continue

            trades = []
            in_position = False
            entry_price = None
            sl_price = None

            for i in range(20, len(df)-1):
                row = df.iloc[i]
                next_row = df.iloc[i+1]
                close = row["close"]
                atr = df["close"].rolling(14).std().iloc[i]

                # Simple Entry Rule: Close > SMA20
                sma20 = df["close"].rolling(20).mean().iloc[i]
                if not in_position and close > sma20:
                    # Compute quantity by risk per trade
                    risk_amount = equity * (self.risk_per_trade_pct / 100)
                    stop_loss = close - atr * 1.5
                    per_share_risk = close - stop_loss
                    quantity = int(risk_amount / per_share_risk)

                    if quantity <= 0:
                        continue

                    entry_price = close * (1 + self.slippage_pct)
                    sl_price = stop_loss
                    in_position = True
                    entry_date = row.name

                elif in_position:
                    # Check exit: SL hit or Close below SMA20
                    exit_reason = None
                    if next_row["low"] <= sl_price:
                        exit_price = sl_price * (1 - self.slippage_pct)
                        exit_reason = "Stop Loss"
                    elif next_row["close"] < sma20:
                        exit_price = next_row["close"] * (1 - self.slippage_pct)
                        exit_reason = "Signal Exit"

                    if exit_reason:
                        pnl = (exit_price - entry_price) * quantity - self.commission_per_trade
                        equity += pnl

                        trades.append({
                            "symbol": symbol,
                            "entry_date": entry_date,
                            "exit_date": next_row.name,
                            "entry_price": round(entry_price,2),
                            "exit_price": round(exit_price,2),
                            "quantity": quantity,
                            "pnl": round(pnl,2),
                            "reason": exit_reason
                        })

                        in_position = False

            # Save results
            if trades:
                trade_df = pd.DataFrame(trades)
                trade_df.to_csv(f"backtest_{symbol}.csv", index=False)
                win_rate = (trade_df["pnl"] > 0).mean() * 100
                avg_pnl = trade_df["pnl"].mean()
                print(f"✅ {symbol}: {len(trade_df)} trades | Win Rate {win_rate:.1f}% | Avg PnL {avg_pnl:.2f}")
                self.results.extend(trades)

        # Summary
        self.export_summary()

    def export_summary(self):
        if not self.results:
            print("⚠️ No trades executed.")
            return
        df = pd.DataFrame(self.results)
        df.to_csv("backtest_summary.csv", index=False)
        print("🎯 Backtesting complete. Summary saved to backtest_summary.csv.")
