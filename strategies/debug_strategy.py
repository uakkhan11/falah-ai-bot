import backtrader as bt

class DebugStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema10 = bt.indicators.EMA(self.data.close, period=10)
        self.ema21 = bt.indicators.EMA(self.data.close, period=21)

    def next(self):
        date = self.datetime.date(0)
        close_price = self.data.close[0]
        rsi_val = self.rsi[0]
        ema10_val = self.ema10[0]
        ema21_val = self.ema21[0]

        print(f"{date} | Close: {close_price:.2f} | RSI: {rsi_val:.2f} | EMA10: {ema10_val:.2f} | EMA21: {ema21_val:.2f}")

        if not self.position:
            reasons = []
            if rsi_val >= 35:
                reasons.append(f"RSI {rsi_val:.2f} >= 35")
            if ema10_val <= ema21_val:
                reasons.append(f"EMA10 {ema10_val:.2f} <= EMA21 {ema21_val:.2f}")
            if reasons:
                print(f"{date} ❌ Skipping BUY | Reasons: {', '.join(reasons)}")
            else:
                print(f"{date} ✅ BUY triggered")
                self.buy()
        else:
            if rsi_val > 70 or ema10_val < ema21_val:
                print(f"{date} ✅ SELL triggered | RSI={rsi_val:.2f}, EMA10={ema10_val:.2f}, EMA21={ema21_val:.2f}")
                self.close()
