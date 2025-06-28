from utils import get_halal_list

sheet_key = "YOUR_SHEET_ID_HERE"
symbols = get_halal_list(sheet_key)
print("✅ Symbols loaded:", symbols)
