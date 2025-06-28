import gspread

gc = gspread.service_account(filename="/root/falah-ai-bot/falah-credentials.json")

# This is your spreadsheet key (just the ID)
SPREADSHEET_KEY = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"

sheet = gc.open_by_key(SPREADSHEET_KEY)

# List all worksheets
worksheets = sheet.worksheets()
for ws in worksheets:
    print("✅ Found worksheet:", ws.title)

# Read some data
first_ws = worksheets[0]
rows = first_ws.get_all_values()
print("✅ First 5 rows:", rows[:5])
