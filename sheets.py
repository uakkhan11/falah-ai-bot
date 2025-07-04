import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import toml

# Load secrets
secrets = toml.load("/root/falah-ai-bot/.streamlit/secrets.toml")
SPREADSHEET_KEY = secrets["sheets"]["SPREADSHEET_KEY"]

def log_exit_to_sheet(stock_name, exit_price, reason, score, timestamp=None):
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # If reason is a list, join into a string
    if isinstance(reason, list):
        reason = ", ".join(reason)

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "/root/falah-ai-bot/falah-credentials.json", scope)
    client = gspread.authorize(creds)

    def log_scan_to_sheet(df):
    gc = gspread.service_account(filename="/root/falah-credentials.json")
    sh = gc.open("FalahSheet")
    ws = sh.worksheet("ScanLog")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        [now, row["Symbol"], row["CMP"], row["Score"], row["Reasons"]]
        for _, row in df.iterrows()
    ]
    ws.append_rows(rows, value_input_option="RAW")

    sheet = client.open_by_key(SPREADSHEET_KEY)
    worksheet = sheet.worksheet("MonitoredStocks")

    data = [timestamp, stock_name, exit_price, reason, score]
    worksheet.append_row(data, value_input_option="USER_ENTERED")
    print(f"✅ Logged exit for {stock_name} to sheet.")
