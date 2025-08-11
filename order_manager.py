import logging
from datetime import datetime

class OrderManager:
    def __init__(self, kite, config):
        self.kite = kite
        self.config = config
        self.positions = {}
        self.orders = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def place_buy_order(self, symbol, quantity, price=None):
        try:  # <- line 24
            # These lines MUST be indented (4 spaces or 1 tab) after the try:
            order_type = self.kite.ORDER_TYPE_MARKET if price is None else self.kite.ORDER_TYPE_LIMIT
            
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                quantity=quantity,
                product=self.kite.PRODUCT_MIS,
                order_type=order_type,
                price=price
            )
            
            self.logger.info(f"✅ Buy order placed: {symbol} qty={quantity} order_id={order_id}")
            return order_id
        
        except Exception as e:
            self.logger.error(f"Order placement failed for {symbol}: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        try:
            positions = self.kite.positions()
            return positions['net']
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            return []
    
    def update_order_status(self):
        """Update order status"""
        try:
            orders = self.kite.orders()
            for order in orders:
                if order['order_id'] in self.orders:
                    self.orders[order['order_id']]['status'] = order['status']
        except Exception as e:
            self.logger.error(f"Failed to update order status: {e}")
