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

    # Convert reason and score to string if needed
    if isinstance(reason, list):
        reason = ", ".join(reason)
    if isinstance(score, list):
        score = ", ".join(str(s) for s in score)

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "/root/falah-ai-bot/falah-credentials.json", scope)
    client = gspread.authorize(creds)

    sheet = client.open_by_key(SPREADSHEET_KEY)
    worksheet = sheet.worksheet("MonitoredStocks")

    data = [timestamp, stock_name, exit_price, reason, score]
    worksheet.append_row(data, value_input_option='USER_ENTERED')

    print(f"✅ Exit logged for {stock_name}")
