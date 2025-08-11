# exit_manager.py
from datetime import datetime
import logging

class ExitManager:
    def __init__(self, config, data_manager, order_manager, trade_logger, notifier):
        self.config = config
        self.data_manager = data_manager
        self.order_manager = order_manager
        self.trade_logger = trade_logger
        self.notifier = notifier
        self.logger = logging.getLogger(__name__)
        self.regime_fail_count = {}  # track consecutive fails

    def check_and_exit_positions(self, positions):
        """
        Apply exit logic to each open position.
        `positions` is a list of dict from OrderTracker.get_positions_with_pl()
        """
        for pos in positions:
            symbol = pos['symbol']
            qty = pos['qty']
            if qty <= 0:
                continue

            # Fetch last N bars for exit logic timeframe (15m or day)
            df = self.data_manager.get_historical_data(symbol, interval="15minute", days=5)
            if df is None or df.empty:
                continue

            df = self._add_exit_indicators(df)
            latest = df.iloc[-1]

            entry_price = pos['avg_price']
            current_price = pos['last_price']
            ret = (current_price - entry_price) / entry_price
            atr_stop = entry_price - self.config.ATR_SL_MULT * latest['atr']

            # Trail logic
            if 'high_since' not in pos:
                pos['high_since'] = entry_price
                pos['trail_active'] = False
                pos['trail_stop'] = 0

            if current_price > pos['high_since']:
                pos['high_since'] = current_price

            if not pos['trail_active'] and ret >= self.config.TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = latest['chandelier_exit']

            if pos['trail_active'] and latest['chandelier_exit'] > pos['trail_stop']:
                pos['trail_stop'] = latest['chandelier_exit']

            # Regime check
            regime_ok = (latest['close'] > latest['ema200']) and (latest['adx'] > self.config.ADX_THRESHOLD_DEFAULT)
            key = f"{symbol}"
            if not regime_ok:
                self.regime_fail_count[key] = self.regime_fail_count.get(key, 0) + 1
            else:
                self.regime_fail_count[key] = 0

            # Exit Decision
            reason = None
            if ret >= self.config.PROFIT_TARGET:
                reason = 'Profit Target'
            elif current_price <= atr_stop:
                reason = 'ATR Stop Loss'
            elif pos['trail_active'] and current_price <= pos['trail_stop']:
                reason = 'Chandelier Exit'
            elif self.regime_fail_count.get(key, 0) >= 2:
                reason = 'Regime Exit'

            if reason:
                order_id = self.order_manager.place_sell_order(symbol, qty, price=current_price)
                self.trade_logger.log_trade(symbol, "SELL", qty, current_price, status="EXIT_"+reason.replace(" ","_"))
                self.notifier.send_message(f"🚪 Exit {symbol} ({reason}) @ ₹{current_price:.2f}")
                self.logger.info(f"Exited {symbol} for reason: {reason}, qty={qty}, price={current_price}")

    def _add_exit_indicators(self, df):
        """Adds the same indicators used in backtest for exit logic."""
        from strategy_utils import add_indicators
        df = add_indicators(df)  # already calculates ATR, EMA200, Chandelier Exit, ADX
        return df
