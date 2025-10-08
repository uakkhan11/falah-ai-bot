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

    def send_text(self, text):
        """
        Synchronous wrapper to send a message from non-async code.
        Safe to call from orchestrator without an event loop.
        """
        if not self.bot:
            self.logger.warning("No bot instance. Message cannot be sent.")
            return False
        try:
            import asyncio
            async def _run():
                try:
                    await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode="HTML")
                    return True
                except TelegramError as e:
                    self.logger.error(f"Telegram send failed: {e}")
                    return False
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(_run())
                    return True
                else:
                    return loop.run_until_complete(_run())
            except RuntimeError:
                # No running loop
                return asyncio.run(__run())
        except Exception as e:
            self.logger.error(f"Telegram sync send failed: {e}")
            return False

    async def send_message(self, text):
        if not self.bot:
            self.logger.warning("No bot instance. Message cannot be sent.")
            return
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode="HTML")
        except TelegramError as e:
            self.logger.error(f"Telegram send failed: {e}")

    async def send_trade_alert(self, symbol, action, qty, price, status):
        msg = (
            f"📢 <b>{action} ALERT</b>\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Qty: {qty}\n"
            f"Price: ₹{price}\n"
            f"Status: {status}"
        )
        await self.send_message(msg)

    async def send_pnl_update(self, positions_with_age):
        if not positions_with_age:
            await self.send_message("💼 No current positions.")
            return
        lines = ["📊 <b>Portfolio P&L Update</b>"]
        for p in positions_with_age:
            lines.append(f"{p['symbol']}: Qty={p['qty']}, PnL=₹{p['pnl']:.2f}, Status={p['holding_status']}")
        await self.send_message("\n".join(lines))

    async def send_t1_t2_change(self, symbol, new_status):
        msg = (
            f"📅 <b>Settlement Status Change</b>\n"
            f"{symbol} → {new_status}"
        )
        await self.send_message(msg)
