# telegram_notifier.py

# telegram_notifier.py
import logging
from telegram import Bot
from telegram.error import TelegramError

class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.logger = logging.getLogger(__name__)
        self.chat_id = chat_id
        try:
            self.bot = Bot(token=bot_token)
        except Exception as e:
            self.logger.error(f"Telegram bot init failed: {e}")
            self.bot = None

    def send_message(self, text):
        """Send a plain text message to the configured Telegram chat."""
        if not self.bot:
            return
        try:
            self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode="HTML")
        except TelegramError as e:
            self.logger.error(f"Telegram send failed: {e}")

    def send_trade_alert(self, symbol, action, qty, price, status):
        msg = f"📢 <b>{action} ALERT</b>\n" \
              f"Symbol: <b>{symbol}</b>\n" \
              f"Qty: {qty}\n" \
              f"Price: ₹{price}\n" \
              f"Status: {status}"
        self.send_message(msg)

    def send_pnl_update(self, positions_with_age):
        if not positions_with_age:
            self.send_message("💼 No current positions.")
            return
        lines = ["📊 <b>Portfolio P&L Update</b>"]
        for p in positions_with_age:
            lines.append(f"{p['symbol']}: Qty={p['qty']}, PnL=₹{p['pnl']:.2f}, Status={p['holding_status']}")
        self.send_message("\n".join(lines))

    def send_t1_t2_change(self, symbol, new_status):
        msg = f"📅 <b>Settlement Status Change</b>\n" \
              f"{symbol} → {new_status}"
        self.send_message(msg)
