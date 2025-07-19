import os
import pandas as pd
import backtrader as bt
from datetime import datetime, timedelta, timezone
from indicators import add_indicators
from ai_engine import get_ai_score

DATA_FOLDER = '/root/falah-ai-bot/historical_data/'

# 6 months lookback for data loading
start_date = datetime.now(timezone.utc) - timedelta(days=180)
filter_start_date = datetime.now(timezone.utc) - timedelta(days=90)

symbol_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
print(f"✅ Total symbols found: {len(symbol_files)}")

valid_symbols = []
dataframes = {}

for file in symbol_files:
    symbol = file.replace('.csv', '')
    try:
        df = pd.read_csv(os.path.join(DATA_FOLDER, file))
        if not {'date', 'open', 'high', 'low', 'close', 'volume'}.issubset(df.columns):
            print(f"⚠️ Skipping {file}: Missing required columns")
            continue

        df.rename(columns={"date": "datetime"}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.sort_values('datetime')

        df = df[df['datetime'] >= start_date]
        if len(df) < 30:
            print(f"⚠️ Skipping {symbol}: Not enough rows after filtering")
            continue

        df = add_indicators(df)
        dataframes[symbol] = df
        valid_symbols.append(symbol)

    except Exception as e:
        print(f"⚠️ Skipping {file}: Error -> {e}")

print(f"✅ Valid symbols loaded: {len(valid_symbols)}")

class AIStrategy(bt.Strategy):
    def __init__(self):
        self.symbol = None
        self.dataclose = self.datas[0].close

    def next(self):
        dt = self.datas[0].datetime.datetime(0).replace(tzinfo=timezone.utc)
        if dt < filter_start_date:
            return  # Only consider last 3 months

        close = self.dataclose[0]
        df = dataframes[self.symbol]
        current_row = df[df['datetime'] == dt]

        if current_row.empty:
            return

        current_row = current_row.iloc[0]
        ai_score = get_ai_score(df[df['datetime'] <= dt].copy())

        if ai_score > 0.25:
            print(f"✅ {self.symbol} BUY {dt.date()} Close={close:.2f} AI={ai_score:.2f}")
            self.buy()

cerebro = bt.Cerebro()
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

for symbol in valid_symbols:
    df = dataframes[symbol].copy()
    df = df[df['datetime'] >= start_date]
    df.set_index('datetime', inplace=True)
    data_feed = bt.feeds.PandasData(dataname=df)

    cerebro.adddata(data_feed, name=symbol)

    strat = cerebro.addstrategy(AIStrategy)
    strat.symbol = symbol

print(f"✅ Starting backtest on {len(valid_symbols)} symbols")
results = cerebro.run()

print("===== BACKTEST COMPLETE =====")
