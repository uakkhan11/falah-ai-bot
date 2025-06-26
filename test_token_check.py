from kiteconnect import KiteConnect
import toml

secrets = toml.load("/root/falah-ai-bot/.streamlit/secrets.toml")
kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])
kite.set_access_token(secrets["zerodha"]["access_token"])

try:
    profile = kite.profile()
    print(f"✅ Logged in as: {profile['user_name']}")
except Exception as e:
    print("❌ Token test failed:", e)
