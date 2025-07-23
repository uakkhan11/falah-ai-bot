# sheets.py
import gspread
import datetime
from credentials import load_secrets

# Load Google Sheets credentials and sheet info
secrets = load_secrets()
GOOGLE_CREDS = "/root/falah-ai-bot/falah-credentials.json"
SHEET_NAME = secrets.get("google", {}).get("sheet_name", "FalahSheet")

def log_trade_to_sheet(symbol, qty, price, rsi, atr, adx, ai_score,
                       action="BUY", exit_reason="", pnl="", outcome=""):
    """
    Log a trade entry to the TradeLog sheet.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gc = gspread.service_account(filename=GOOGLE_CREDS)
    sh = gc.open(SHEET_NAME)
    ws = sh.worksheet("TradeLog")

    ws.append_row([
        timestamp,
        symbol,
        qty,
        price,
        "",         # Exit price is empty for new entries
        rsi,
        atr,
        adx,
        ai_score,
        action,
        exit_reason,
        pnl,
        outcome
    ], value_input_option="USER_ENTERED")

    print(f"✅ Logged trade for {symbol}.")

def log_exit_to_sheet(sheet_name, worksheet_name, symbol, exit_price, reason):
    """
    Log an exit trade to a custom worksheet.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gc = gspread.service_account(filename=GOOGLE_CREDS)
    sh = gc.open(sheet_name)
    ws = sh.worksheet(worksheet_name)

    ws.append_row([
        timestamp,
        symbol,
        exit_price,
        reason
    ], value_input_option="USER_ENTERED")

    print(f"✅ Logged exit for {symbol}")

def log_scan_to_sheet(df):
    """
    Log a DataFrame of scan results to the ScanLog sheet.
    """
    gc = gspread.service_account(filename=GOOGLE_CREDS)
    sh = gc.open(SHEET_NAME)
    ws = sh.worksheet("ScanLog")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = [
        [now, row["Symbol"], row["CMP"], row["Score"], row["Reasons"]]
        for _, row in df.iterrows()
    ]

    ws.append_rows(rows, value_input_option="RAW")
    print(f"✅ Logged {len(rows)} scan results to sheet.")
