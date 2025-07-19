import backtrader as bt

class AlwaysBuyStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)

    def next(self):
        date = self.datetime.date(0)
        rsi_val = self.rsi[0]
        print(f"{date} | RSI: {rsi_val:.2f}")

        if not self.position and len(self) > 20:
            print(f"{date} ✅ Always BUY after indicators ready")
            self.buy()
        elif self.position and rsi_val > 70:
            print(f"{date} ✅ SELL as RSI > 70")
            self.close()
