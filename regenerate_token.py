from kiteconnect import KiteConnect
import toml

# Load secrets from .streamlit/secrets.toml
with open("/root/falah-ai-bot/.streamlit/secrets.toml", "r") as f:
    secrets = toml.load(f)

api_key = secrets["zerodha"]["api_key"]
api_secret = secrets["zerodha"]["api_secret"]
request_token = secrets["zerodha"]["request_token"]  # You must update this manually each day

kite = KiteConnect(api_key=api_key)

try:
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]
    print("✅ New Access Token:", access_token)

    # Update secrets.toml file
    secrets["zerodha"]["access_token"] = access_token
    with open("/root/falah-ai-bot/.streamlit/secrets.toml", "w") as f:
        toml.dump(secrets, f)

    print("✅ secrets.toml updated successfully.")
except Exception as e:
    print("❌ Error refreshing token:", e)
