# sheets.py

import gspread
import datetime

def log_exit_to_sheet(stock_name, exit_price, reason, score, timestamp=None):
    """
    Log a stock exit event to the MonitoredStocks sheet.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(reason, list):
        reason = ", ".join(reason)

    gc = gspread.service_account(filename="/root/falah-ai-bot/falah-credentials.json")

    sh = gc.open("FalahSheet")
    ws = sh.worksheet("MonitoredStocks")

    data = [timestamp, stock_name, exit_price, reason, score]
    ws.append_row(data, value_input_option="USER_ENTERED")
    print(f"✅ Logged exit for {stock_name} to sheet.")


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
