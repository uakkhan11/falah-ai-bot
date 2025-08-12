# fetch_token.py

import json
from kiteconnect import KiteConnect

# 1️⃣  Your Zerodha API credentials
API_KEY = "ijzeuwuylr3g0kug"
API_SECRET = "yy1wd2wn8r0wx4mus00vxllgss03nuqx"

# File to save the generated access token
TOKENS_FILE = "A_tokens.json"

def main():
    kite = KiteConnect(api_key=API_KEY)

    # 2️⃣  Generate login URL
    login_url = kite.login_url()
    print("\n🔗 LOGIN URL (open in browser):\n", login_url)
    print("\n1. Log in to Zerodha Kite.")
    print("2. Complete 2FA.")
    print("3. You'll be redirected to your set Redirect URL.")
    print("4. Copy the value after 'request_token=' from the browser address bar.\n")

    # 3️⃣  Prompt for request_token
    request_token = input("Paste request_token here: ").strip()

    # 4️⃣  Exchange request_token for access_token
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        print("\n✅ Access token obtained:", access_token)

        # 5️⃣  Save to file
        with open(TOKENS_FILE, "w") as f:
            json.dump({"access_token": access_token}, f)
        print(f"💾 Access token saved to {TOKENS_FILE}")

    except Exception as e:
        print("\n❌ Error generating access token:", e)

if __name__ == "__main__":
    main()
