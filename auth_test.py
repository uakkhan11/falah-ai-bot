# auth_test.py
import json
import os
from kiteconnect import KiteConnect

TOKENS_FILE = "kite_tokens.json"

# ===== CONFIGURE THESE =====
API_KEY = "your_api_key_here"
API_SECRET = "your_api_secret_here"
# ===========================

def save_access_token(token):
    with open(TOKENS_FILE, "w") as f:
        json.dump({"access_token": token}, f)
    print(f"✅ Access token saved to {TOKENS_FILE}")

def load_access_token():
    if os.path.exists(TOKENS_FILE):
        with open(TOKENS_FILE, "r") as f:
            data = json.load(f)
            return data.get("access_token")
    return None

def authenticate():
    kite = KiteConnect(api_key=API_KEY)

    # Step 1: Check if already have token
    access_token = load_access_token()
    if access_token:
        kite.set_access_token(access_token)
        try:
            profile = kite.profile()
            print("✅ Already authenticated! Profile:", profile["user_name"])
            return
        except Exception as e:
            print("⚠️ Saved token invalid, doing fresh login...", e)

    # Step 2: Fresh login
    login_url = kite.login_url()
    print(f"\n🔑 Login URL:\n{login_url}\n")
    print("1. Open the above URL in a browser")
    print("2. Login & complete 2FA")
    print("3. Copy the `request_token` from the redirected URL\n")

    request_token = input("Paste request_token here: ").strip()

    # Step 3: Exchange for access_token
    try:
        session_data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = session_data["access_token"]
        kite.set_access_token(access_token)
        save_access_token(access_token)
        print("✅ Authentication successful!")
        print("👤 Profile:", kite.profile()["user_name"])
    except Exception as e:
        print("❌ Authentication failed:", e)

if __name__ == "__main__":
    authenticate()
