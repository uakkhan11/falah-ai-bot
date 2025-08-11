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
        """Place buy order with error handling"""
        try:
            # Get current margin
            margins = self.kite.margins()
            available_cash = float(margins['equity']['available']['cash'])
            
            order_value = quantity * (price or self.get_current_price(symbol))
            
            if available_cash < order_value:
                self.logger.warning(f"Insufficient funds for {symbol}")
                return None
                
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                quantity=quantity,
                product=self.kite.PRODUCT_MIS,  # Intraday
                order_type=self.kite.ORDER_TYPE_MARKET if not price else self.kite.ORDER_TYPE_LIMIT,
                price=price
            )
            
            self.orders[order_id] = {
                'symbol': symbol,
                'quantity': quantity,
                'side': 'BUY',
                'timestamp': datetime.now(),
                'status': 'PENDING'
            }
            
            self.logger.info(f"Buy order placed: {symbol} qty={quantity} order_id={order_id}")
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
