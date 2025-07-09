# backtester.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, 
                 data_dir,
                 symbols,
                 initial_capital=100000,
                 slippage_pct=0.002,
                 commission_per_trade=20,
                 risk_per_trade_pct=2,
                 walk_forward_train_days=90,
                 walk_forward_test_days=30,
                 strategy_func=None):
        """
        strategy_func: a function accepting df, i and returning (entry_signal, exit_signal)
        """
        self.data_dir = data_dir
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_per_trade = commission_per_trade
        self.risk_per_trade_pct = risk_per_trade_pct
        self.walk_forward_train_days = walk_forward_train_days
        self.walk_forward_test_days = walk_forward_test_days
        self.strategy_func = strategy_func
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

    def compute_metrics(self, equity_curve):
        returns = equity_curve.pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        drawdown = (equity_curve / equity_curve.cummax() - 1).min()
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1
        return sharpe, drawdown, cagr

    def run(self):
        equity = self.initial_capital
        equity_curve = []

        for symbol in self.symbols:
            df = self.load_data(symbol)
            if df is None or len(df) < (self.walk_forward_train_days + self.walk_forward_test_days):
                continue

            trades = []
            i = 0

            while i + self.walk_forward_train_days + self.walk_forward_test_days <= len(df):
                train_df = df.iloc[i : i + self.walk_forward_train_days]
                test_df = df.iloc[i + self.walk_forward_train_days : i + self.walk_forward_train_days + self.walk_forward_test_days]

                i += self.walk_forward_test_days

                in_position = False
                entry_price = None
                sl_price = None

                for j in range(len(test_df)-1):
                    idx = train_df.shape[0] + j
                    row = df.iloc[idx]
                    next_row = df.iloc[idx+1]
                    close = row["close"]
                    atr = df["close"].rolling(14).std().iloc[idx]

                    if not self.strategy_func:
                        sma20 = df["close"].rolling(20).mean().iloc[idx]
                        entry_signal = close > sma20
                        exit_signal = close < sma20
                    else:
                        entry_signal, exit_signal = self.strategy_func(df, idx)

                    if not in_position and entry_signal:
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
                        exit_reason = None
                        if next_row["low"] <= sl_price:
                            exit_price = sl_price * (1 - self.slippage_pct)
                            exit_reason = "Stop Loss"
                        elif exit_signal:
                            exit_price = next_row["close"] * (1 - self.slippage_pct)
                            exit_reason = "Signal Exit"

                        if exit_reason:
                            pnl = (exit_price - entry_price) * quantity - self.commission_per_trade
                            equity += pnl
                            equity_curve.append(equity)
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

            if trades:
                trade_df = pd.DataFrame(trades)
                trade_df.to_csv(f"backtest_{symbol}.csv", index=False)
                win_rate = (trade_df["pnl"] > 0).mean() * 100
                avg_pnl = trade_df["pnl"].mean()
                print(f"✅ {symbol}: {len(trade_df)} trades | Win Rate {win_rate:.1f}% | Avg PnL {avg_pnl:.2f}")
                self.results.extend(trades)

        # Final Equity Curve
        if equity_curve:
            eq_series = pd.Series(equity_curve)
            sharpe, drawdown, cagr = self.compute_metrics(eq_series)
            eq_series.plot(title="Equity Curve")
            plt.savefig("equity_curve.png")
            print(f"📊 Sharpe: {sharpe:.2f} | Max Drawdown: {drawdown:.2%} | CAGR: {cagr:.2%}")

        self.export_summary()

    def export_summary(self):
        if not self.results:
            print("⚠️ No trades executed.")
            return
        df = pd.DataFrame(self.results)
        df.to_csv("backtest_summary.csv", index=False)
        print("🎯 Backtesting complete. Summary saved to backtest_summary.csv.")
