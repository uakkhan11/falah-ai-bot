import asyncio
import pandas as pd
from unittest.mock import AsyncMock, MagicMock

from bot_logic import FalahTradingBot
from strategy_utils import add_indicators

# Create a dry-run mock OrderManager (no real orders)
class MockOrderManager:
    def __init__(self):
        self.placed_orders = []

    def place_buy_order(self, symbol, qty, price=None):
        self.placed_orders.append(('BUY', symbol, qty, price))
        return f"MOCK_BUY_ORDER_{symbol}"

    def place_sell_order(self, symbol, qty, price=None):
        self.placed_orders.append(('SELL', symbol, qty, price))
        return f"MOCK_SELL_ORDER_{symbol}"

# Create a dry-run mock TelegramNotifier (no real messages)
class MockTelegramNotifier:
    async def send_trade_alert(self, symbol, side, qty, price, status):
        print(f"Telegram Alert: {side} {qty} {symbol} @ {price} - {status}")

    async def send_message(self, message):
        print(f"Telegram Msg: {message}")

# Sample function to load and prepare test data for a symbol
def load_test_data(symbol):
    df_daily = pd.read_csv(f"swing_data/{symbol}.csv", parse_dates=['date'])
    df_15m = pd.read_csv(f"scalping_data/{symbol}.csv", parse_dates=['date'])
    df_daily = df_daily.sort_values('date').reset_index(drop=True)
    df_15m = df_15m.sort_values('date').reset_index(drop=True)
    df_daily = add_indicators(df_daily)
    df_15m = add_indicators(df_15m)
    return df_daily, df_15m

async def dry_run_test():
    bot = FalahTradingBot()

    # Inject mocks
    bot.order_manager = MockOrderManager()
    bot.notifier = MockTelegramNotifier()
    bot.trade_logger = MagicMock()
    bot.capital_manager = MagicMock()
    bot.risk_manager = MagicMock()
    bot.exit_manager = MagicMock()
    bot.live_price_streamer = MagicMock()
    bot.live_candle_aggregator = MagicMock()
    bot.order_tracker = MagicMock()

    bot.authenticated = True
    bot.trading_symbols = ['RELIANCE']  # Test with one symbol for quick testing
    bot.instruments = {'RELIANCE': 256265}  # Mock token

    # Setup capital and risk mocks
    bot.capital_manager.get_available_capital.return_value = 1000000
    bot.capital_manager.adjust_quantity_for_capital.return_value = (100, None)
    bot.capital_manager.allocate_capital.return_value = None
    bot.capital_manager.free_capital.return_value = None
    bot.risk_manager.allow_trade.return_value = (True, None)

    # Setup live candle data mock
    df_daily, df_15m = load_test_data('RELIANCE')
    # Provide latest candle for live_candle_aggregator.get_all_live_candles
    latest_candle = {
        'ts': df_15m.iloc[-1]['date'],
        'open': df_15m.iloc[-1]['open'],
        'high': df_15m.iloc[-1]['high'],
        'low': df_15m.iloc[-1]['low'],
        'close': df_15m.iloc[-1]['close'],
        'volume': df_15m.iloc[-1]['volume']
    }
    bot.live_candle_aggregator.get_all_live_candles.return_value = {256265: latest_candle}
    # Provide live price streamer price
    bot.live_price_streamer.get_price.return_value = df_15m.iloc[-1]['close']
    # Provide order_tracker positions (empty at start)
    bot.order_tracker.get_positions_with_pl.return_value = []

    # Patch methods to use test data csv contents inside run_cycle
    # We'll monkey patch pandas read_csv here for quick override in dry run
    original_read_csv = pd.read_csv

    def mock_read_csv(filepath, *args, **kwargs):
        if f"swing_data/RELIANCE.csv" in filepath:
            return df_daily
        elif f"scalping_data/RELIANCE.csv" in filepath:
            return df_15m
        else:
            return original_read_csv(filepath, *args, **kwargs)

    pd.read_csv = mock_read_csv

    # Run one cycle
    result = bot.run_cycle()
    print("Run cycle result:")
    print(result)

    # Restore original pd.read_csv
    pd.read_csv = original_read_csv

# Run the dry run test
asyncio.run(dry_run_test())
