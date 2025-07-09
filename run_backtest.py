from backtester import Backtester
from strategies import sma_strategy

bt = Backtester(
    data_dir="/root/falah-ai-bot/historical_data",
    symbols=["INFY","TCS","HDFCBANK"],
    initial_capital=100000,
    slippage_pct=0.002,
    commission_per_trade=20,
    risk_per_trade_pct=2,
    walk_forward_train_days=90,
    walk_forward_test_days=30,
    strategy_func=sma_strategy
)

bt.run()
