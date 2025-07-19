import os
import pandas as pd
import backtrader as bt
from datetime import datetime, timedelta
from indicators import add_indicators
from ai_engine import get_ai_score

# === Configuration ===
data_folder = "/root/falah-ai-bot/historical_data"
start_date = datetime.now() - timedelta(days=90)
min_rows_required = 60
ai_score_threshold = 0.25

# === Helper function to validate CSV ===
def is_valid_csv(df):
    required_columns = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
    return required_columns.issubset(df.columns)

# === Backtrader Strategy ===
class RecentAIStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        date = self.datas[0].datetime.date(0)
        close = self.dataclose[0]
        rsi = self.datas[0].rsi[0]
        ema10 = self.datas[0].ema10[0]
        ema21 = self.datas[0].ema21[0]
        ai_score = self.datas[0].ai_score[0]

        if not self.position:
            if (rsi > 35 and rsi < 65) and (ema10 > ema21) and (ai_score >= ai_score_threshold):
                print(f"✅ BUY | {self.datas[0]._name} | {date} | Close={close:.2f}, RSI={rsi:.2f}, AI={ai_score:.2f}")
                self.buy()

# === Load valid symbols ===
symbol_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
print(f"✅ Total files found: {len(symbol_files)}")

valid_symbols = []

for file in symbol_files:
    file_path = os.path.join(data_folder, file)
    try:
        df = pd.read_csv(file_path)
        if 'date' in df.columns:
            df.rename(columns={'date': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)
        df = df[df['datetime'] >= pd.to_datetime(start_date).tz_localize('Asia/Kolkata')]

        if len(df) < min_rows_required:
            print(f"⚠️ Skipping {file}: Not enough recent data")
            continue

        if not is_valid_csv(df):
            print(f"⚠️ Skipping {file}: Missing required columns")
            continue

        valid_symbols.append((file.replace('.csv', ''), df))
    except Exception as e:
        print(f"⚠️ Skipping {file}: {e}")

print(f"✅ Total valid symbols (last 3 months): {len(valid_symbols)}")

# === Run Backtest ===
cerebro = bt.Cerebro()

for symbol, df in valid_symbols:
    df = add_indicators(df)
    df['ai_score'] = df.apply(get_ai_score, axis=1)

    btfeed = bt.feeds.PandasData(
        dataname=df,
        datetime='datetime',
        open='open', high='high', low='low', close='close', volume='volume',
        openinterest=-1,
        rsi='rsi', ema10='ema10', ema21='ema21', ai_score='ai_score'
    )
    cerebro.adddata(btfeed, name=symbol)

cerebro.addstrategy(RecentAIStrategy)
cerebro.run()
print("===== BACKTEST COMPLETE =====")
