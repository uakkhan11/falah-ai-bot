import backtrader as bt

class DebugStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema10 = bt.indicators.EMA(self.data.close, period=10)
        self.ema21 = bt.indicators.EMA(self.data.close, period=21)

    def log(self, txt):
        date = self.datas[0].datetime.date(0)
        print(f"{date} | {txt}")

    def next(self):
        date = self.datas[0].datetime.date(0)
        close = self.data.close[0]
        rsi = self.rsi[0]
        ema10 = self.ema10[0]
        ema21 = self.ema21[0]

        self.log(f"Close: {close:.2f} | RSI: {rsi:.2f} | EMA10: {ema10:.2f} | EMA21: {ema21:.2f}")

        if not self.position:
            reasons = []
            if rsi > 75:
                reasons.append(f"RSI {rsi:.2f} > 75")
            if ema10 <= ema21:
                reasons.append(f"EMA10 {ema10:.2f} <= EMA21 {ema21:.2f}")

            if reasons:
                self.log(f"❌ Skipping BUY | Reasons: {', '.join(reasons)}")
            else:
                self.log("✅ BUY triggered")
                self.buy()
        else:
            if rsi < 30 or ema10 < ema21:
                self.log(f"✅ SELL triggered | RSI={rsi:.2f}, EMA10={ema10:.2f}, EMA21={ema21:.2f}")
                self.close()
