# trading_journal_bot.py
# python-telegram-bot v20+
# pip install python-telegram-bot==20.* pandas

import os
import pandas as pd
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, ConversationHandler
)

BOT_TOKEN = "7763450358:AAH32bWYyu_hXR6l-UaVMaarFGZ4YFOv6q8"  # replace
DATA_DIR = "/root/falah-ai-bot"
CSV_PATH = os.path.join(DATA_DIR, "trades.csv")

# Expected schema for safe pandas operations
REQUIRED_COLUMNS = [
    "chat_id", "timestamp", "symbol", "side", "qty", "price", "notes"
]

# Simple per-chat state for conversation inputs
ASK_SYMBOL, ASK_SIDE, ASK_QTY, ASK_PRICE, ASK_NOTES = range(5)

def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        df.to_csv(CSV_PATH, index=False)

def load_df():
    ensure_storage()
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        # Corrupt or unreadable file fallback
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    # Normalize columns to avoid KeyError
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    # Keep only expected schema
    df = df[REQUIRED_COLUMNS]
    return df

def append_trade(record: dict):
    df = load_df()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to Trading Journal Bot.\nCommands:\n"
        "/add - add a trade\n/list - show last 10 trades\n/export - export all trades to CSV",
    )

async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["trade"] = {}
    await update.message.reply_text("Enter symbol (e.g., BTCUSDT):")
    return ASK_SYMBOL

async def ask_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["trade"]["symbol"] = update.message.text.strip().upper()
    kb = ReplyKeyboardMarkup([["BUY", "SELL"]], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Side?", reply_markup=kb)
    return ASK_SIDE

async def ask_side(update: Update, context: ContextTypes.DEFAULT_TYPE):
    side = update.message.text.strip().upper()
    if side not in ["BUY", "SELL"]:
        await update.message.reply_text("Please choose BUY or SELL.")
        return ASK_SIDE
    context.user_data["trade"]["side"] = side
    await update.message.reply_text("Quantity?", reply_markup=ReplyKeyboardRemove())
    return ASK_QTY

async def ask_qty(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        qty = float(update.message.text.strip())
        if qty <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("Enter a positive number for quantity.")
        return ASK_QTY
    context.user_data["trade"]["qty"] = qty
    await update.message.reply_text("Price?")
    return ASK_PRICE

async def ask_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        price = float(update.message.text.strip())
        if price <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text("Enter a positive number for price.")
        return ASK_PRICE
    context.user_data["trade"]["price"] = price
    await update.message.reply_text("Notes? (or type '-' to skip)")
    return ASK_NOTES

async def ask_notes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    notes = update.message.text.strip()
    if notes == "-":
        notes = ""
    trade = context.user_data.get("trade", {})
    trade["notes"] = notes
    trade["chat_id"] = update.effective_chat.id
    trade["timestamp"] = datetime.utcnow().isoformat()

    append_trade(trade)
    await update.message.reply_text(
        f"Saved: {trade['symbol']} {trade['side']} qty={trade['qty']} price={trade['price']}"
    )
    context.user_data["trade"] = {}
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["trade"] = {}
    await update.message.reply_text("Canceled.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

def fmt_row(row):
    return (
        f"{row['timestamp']} | {row['symbol']} {row['side']} "
        f"qty={row['qty']} price={row['price']} {('- ' + str(row['notes'])) if str(row['notes']).strip() else ''}"
    )

async def list_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = load_df()
    view = df[df["chat_id"] == update.effective_chat.id].tail(10)
    if view.empty:
        await update.message.reply_text("No trades yet.")
        return
    lines = [fmt_row(r) for _, r in view.iterrows()]
    await update.message.reply_text("Last trades:\n" + "\n".join(lines))

async def export_csv(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_storage()
    await update.message.reply_document(document=open(CSV_PATH, "rb"), filename="trades.csv")

def main():
    ensure_storage()
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("add", add)],
        states={
            ASK_SYMBOL: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_symbol)],
            ASK_SIDE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_side)],
            ASK_QTY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_qty)],
            ASK_PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_price)],
            ASK_NOTES: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_notes)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)
    app.add_handler(CommandHandler("list", list_trades))
    app.add_handler(CommandHandler("export", export_csv))

    app.run_polling()

if __name__ == "__main__":
    main()
