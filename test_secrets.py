# test_secrets.py
import toml

with open(".streamlit/secrets.toml", "r") as f:
    secrets = toml.load(f)

print("✅ API Key:", secrets["zerodha"]["api_key"])
print("✅ Access Token:", secrets["zerodha"]["access_token"])
print("✅ Google Sheet Key:", secrets["global"]["google_sheet_key"])

