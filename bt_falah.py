import backtrader as bt

class FalahSanityStrategy(bt.Strategy):
    def __init__(self):
        self.order = None

    def next(self):
        if not self.position:
            # Always buy 1 share
            self.order = self.buy(size=1)
        else:
            # Close after 1 bar
            self.order = self.close()
