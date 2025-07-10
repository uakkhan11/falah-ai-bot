import gspread
import datetime

def log_exit_to_sheet(
    sheet_name,
    worksheet_name,
    symbol,
    exit_price,
    reason
):
    """
    Log an exit trade to the specified sheet.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gc = gspread.service_account(filename="/root/falah-ai-bot/falah-credentials.json")
    sh = gc.open(sheet_name)
    ws = sh.worksheet(worksheet_name)

    ws.append_row([
        timestamp,
        symbol,
        exit_price,
        reason
    ], value_input_option="USER_ENTERED")
    print(f"✅ Logged exit for {symbol}")

def log_trade_to_sheet(
    sheet,
    timestamp,
    symbol,
    quantity,
    entry_price,
    exit_price,
    rsi,
    atr,
    adx,
    ai_score,
    action,
    exit_reason,
    pnl,
    outcome
):
    """
    Log a trade entry to the sheet.
    """
    sheet.append_row([
        timestamp,
        symbol,
        quantity,
        entry_price,
        exit_price,
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

def log_scan_to_sheet(df):
    """
    Log a DataFrame of scan results to the ScanLog sheet.
    """
    gc = gspread.service_account(filename="/root/falah-ai-bot/falah-credentials.json")
    sh = gc.open("FalahSheet")
    ws = sh.worksheet("ScanLog")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = [
        [now, row["Symbol"], row["CMP"], row["Score"], row["Reasons"]]
        for _, row in df.iterrows()
    ]

    ws.append_rows(rows, value_input_option="RAW")
    print(f"✅ Logged {len(rows)} scan results to sheet.")
