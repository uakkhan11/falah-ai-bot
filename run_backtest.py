from backtester import Backtester

bt = Backtester(
    data_dir="/root/falah-ai-bot/historical_data",
    symbols=["INFY", "TCS", "HDFCBANK"],
    initial_capital=100000,
    slippage_pct=0.002,
    commission_per_trade=20,
    risk_per_trade_pct=2
)

bt.run()
