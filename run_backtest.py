import os
import pandas as pd
import joblib
import backtrader as bt
from datetime import datetime

HIST_DIR = "/root/falah-ai-bot/historical_data/"
MODEL_PATH = "/root/falah-ai-bot/model.pkl"

model = joblib.load(MODEL_PATH)

def get_ai_score(rsi, atr, ema10, ema21, volume_change):
    features = pd.DataFrame([{
        "RSI": rsi,
        "EMA10": ema10,
        "EMA21": ema21,
        "ATR": atr,
        "VolumeChange": volume_change
    }])
    prob = model.predict_proba(features)[0][1]
    return prob

class AIStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema10 = bt.indicators.EMA(self.data.close, period=10)
        self.ema21 = bt.indicators.EMA(self.data.close, period=21)
        self.atr = bt.indicators.ATR(self.data, period=14)

    def next(self):
        date = self.data.datetime.date(0)
        rsi_val = self.rsi[0]
        ema10_val = self.ema10[0]
        ema21_val = self.ema21[0]
        atr_val = self.atr[0]
        close = self.data.close[0]
        volume = self.data.volume[0]

        volume_change = 0.0  # You can add proper volume change logic if needed

        ai_score = get_ai_score(rsi_val, atr_val, ema10_val, ema21_val, volume_change)

        print(f"{date} | Close: {close:.2f} | RSI: {rsi_val:.2f} | EMA10: {ema10_val:.2f} | EMA21: {ema21_val:.2f} | AI Score: {ai_score:.2f}")

        if not self.position:
            if 40 <= rsi_val <= 65 and ema10_val > ema21_val and ai_score >= 0.25:
                print(f"{date} | ✅ BUY triggered | AI Score={ai_score:.2f}")
                self.buy()
        else:
            if rsi_val > 70 or ema10_val < ema21_val:
                print(f"{date} | ❌ SELL triggered | RSI={rsi_val:.2f}")
                self.close()

def load_csv(csv_file):
    try:
        df = pd.read_csv(csv_file, parse_dates=["date"])
        df.dropna(subset=["date", "open", "high", "low", "close", "volume"], inplace=True)
        if len(df) < 100:
            print(f"⚠️ Skipping {csv_file}: insufficient data ({len(df)} rows).")
            return None
        df.sort_values("date", inplace=True)
        return df
    except Exception as e:
        print(f"❌ Error loading {csv_file}: {e}")
        return None

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AIStrategy)
    cerebro.broker.set_cash(100000)

    csv_files = [os.path.join(HIST_DIR, f) for f in os.listdir(HIST_DIR) if f.endswith(".csv")]
    print(f"✅ Found {len(csv_files)} CSV files.")

    valid_count = 0
    for csv_file in csv_files:
        df = load_csv(csv_file)
        if df is None:
            continue

        data = bt.feeds.PandasData(
            dataname=df,
            datetime="date", open="open", high="high", low="low",
            close="close", volume="volume", openinterest=None
        )
        cerebro.adddata(data)
        valid_count += 1

    print(f"✅ Loaded {valid_count} valid symbols.")

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print("🚀 Starting backtest...")
    results = cerebro.run()
    strat = results[0]

    print(f"🏁 Final Portfolio Value: ₹{cerebro.broker.getvalue():.2f}")
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"📊 Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")

    trades = strat.analyzers.trades.get_analysis()
    print(f"📈 Total Trades: {trades.total.closed if trades.total.closed else 'N/A'}")
