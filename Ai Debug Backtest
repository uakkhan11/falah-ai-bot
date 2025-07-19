import backtrader as bt
import pandas as pd
import os, json
from indicators import add_indicators
from ai_engine import get_ai_score

class DebugStrategy(bt.Strategy):
    def __init__(self):
        add_indicators(self)

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt}, {txt}')

    def next(self):
        rsi = self.rsi[0]
        ema10 = self.ema10[0]
        ema21 = self.ema21[0]
        close = self.data.close[0]
        ai_score = get_ai_score({
            'rsi': rsi,
            'ema10': ema10,
            'ema21': ema21,
            'close': close,
            'atr': self.atr[0],
            'volume_change': self.volume_change[0]
        })

        reason = []
        if rsi < 35 or rsi > 70:
            reason.append(f"RSI={rsi:.2f}")
        if ema10 < ema21:
            reason.append(f"EMA10={ema10:.2f} < EMA21={ema21:.2f}")
        if ai_score < 0.25:
            reason.append(f"AI_Score={ai_score:.2f}")

        if reason:
            self.log(f"NO TRADE: {', '.join(reason)}")
        else:
            self.log(f"BUY CREATE {close:.2f}, AI_Score={ai_score:.2f}, RSI={rsi:.2f}, EMA10={ema10:.2f}, EMA21={ema21:.2f}")
            self.buy()

if __name__ == "__main__":
    folder = '/root/falah-ai-bot/historical_data/'
    large_mid_cap = json.load(open('/root/falah-ai-bot/large_mid_cap.json'))

    cerebro = bt.Cerebro()
    symbol_count, trade_count = 0, 0

    for file in os.listdir(folder):
        symbol = file.replace('.csv', '')
        if symbol not in large_mid_cap:
            continue
        df = pd.read_csv(folder + file)
        if len(df) < 100:
            continue
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        symbol_count += 1

    cerebro.addstrategy(DebugStrategy)
    cerebro.run()
    print(f"===== FINAL SUMMARY =====\nTotal Symbols Backtested: {symbol_count}\n===== END =====")
