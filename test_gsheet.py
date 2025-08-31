from google.oauth2.service_account import Credentials
import gspread

scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = Credentials.from_service_account_file(
    "/root/falah-ai-bot/falah-credentials.json", scopes=scopes)

client = gspread.authorize(creds)
try:
    sheet = client.open_by_key("1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")
    worksheet = sheet.worksheet("HalalList")
    symbols = worksheet.col_values(1)
    print(symbols[:10])  # preview first 10
except Exception as e:
    print(f"Failed to fetch halal symbols: {e}")
