import gspread
from oauth2client.service_account import ServiceAccountCredentials

CREDS_JSON = "falah-credentials.json"
SHEET_KEY = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_JSON, scope)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_KEY)

print("✅ Sheet Title:", sheet.title)
