# test_append.py
import gspread
gc = gspread.service_account(filename="/root/falah-credentials.json")
sh = gc.open_by_key("1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")
ws = sh.worksheet("MonitoredStocks")
ws.append_row(["2025-06-27", "TESTSTOCK", 10, 100.0, "--", "HOLD"])
print("✅ Row added")
