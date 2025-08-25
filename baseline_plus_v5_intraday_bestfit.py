import pandas as pd
from strategy_utils import add_indicators
from datetime import datetime

ATR_MULT = 2.0

class BacktestBot:
    def __init__(self, symbol, df, initial_capital=100000, timeframe='15m'):
        self.symbol = symbol
        self.df = df
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0  # number of shares held
        self.entry_price = 0
        self.trades = []
        self.timeframe = timeframe

    def run_backtest(self):
        self.df = add_indicators(self.df)

        for i in range(len(self.df)):
            row = self.df.iloc[i]

            # Skip if not enough data for indicators
            if pd.isna(row.get('ema200', None)) or pd.isna(row.get('atr', None)):
                continue

            # Check if holding position: apply exit logic
            if self.position > 0:
                stop_loss = max(row.get('chandelier_exit', 0), self.entry_price - ATR_MULT * row['atr'])
                stop_loss_hit = row['low'] <= stop_loss
                momentum_exit = (row.get('rsi_14', 100) < 70 and row['close'] < row.get('bb_upper', 0)) or (row.get('supertrend_direction', 1) < 0)

                if stop_loss_hit or momentum_exit:
                    exit_price = row['close']
                    self.cash += self.position * exit_price
                    self.trades.append({
                        'type': 'SELL',
                        'price': exit_price,
                        'qty': self.position,
                        'date': row['date'],
                        'pnl': (exit_price - self.entry_price) * self.position
                    })
                    print(f"{self.timeframe} {self.symbol} SELL {self.position} @ {exit_price} on {row['date']} PnL: {(exit_price - self.entry_price) * self.position}")
                    self.position = 0
                    self.entry_price = 0

            # Check entry condition only if not holding
            if self.position == 0:
                daily_up = row['close'] > row.get('ema200', 0)
                entry_signal = row.get('entry_signal', 0) == 1
                if daily_up and entry_signal:
                    # Simple position sizing: buy as many shares as possible with 10% capital per trade
                    risk_per_trade = self.cash * 0.10
                    qty = int(risk_per_trade / row['close'])
                    if qty > 0:
                        self.position = qty
                        self.entry_price = row['close']
                        cost = qty * row['close']
                        self.cash -= cost
                        self.trades.append({
                            'type': 'BUY',
                            'price': row['close'],
                            'qty': qty,
                            'date': row['date']
                        })
                        print(f"{self.timeframe} {self.symbol} BUY {qty} @ {row['close']} on {row['date']}")

        total_value = self.cash + (self.position * self.df.iloc[-1]['close'] if self.position > 0 else 0)
        print(f"Backtest complete for {self.symbol} {self.timeframe}. Initial capital: {self.initial_capital}, Final value: {total_value}")

        return self.trades, total_value

if __name__ == "__main__":
    # Load CSVs for 15m scalping and 1h swing data
    symbol = "RELIANCE"

    df_15m = pd.read_csv(f"scalping_data/{symbol}.csv", parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df_1h = pd.read_csv(f"intraday_swing_data/{symbol}.csv", parse_dates=['date']).sort_values('date').reset_index(drop=True)

    # Run scalping backtest
    bot_15m = BacktestBot(symbol, df_15m, timeframe='15m')
    trades_15m, final_value_15m = bot_15m.run_backtest()

    # Run 1-hour backtest
    bot_1h = BacktestBot(symbol, df_1h, timeframe='1h')
    trades_1h, final_value_1h = bot_1h.run_backtest()
