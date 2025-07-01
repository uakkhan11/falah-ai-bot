# token_loader.py

import json
from kiteconnect import KiteConnect

def load_or_create_token(api_key, api_secret, interactive=True, st=None):
    """
    Tries to load and validate access_token.json.
    If invalid, asks user for request_token and saves new access_token.
    Returns KiteConnect object with token set.
    """
    kite = KiteConnect(api_key=api_key)
    token_path = "/root/falah-ai-bot/access_token.json"
    access_token = None

    # Try to load
    try:
        with open(token_path) as f:
            access_token = json.load(f)["access_token"]
        kite.set_access_token(access_token)
        profile = kite.profile()
        msg = f"✅ Access token valid. Logged in as {profile['user_name']}."
        if interactive and st:
            st.success(msg)
        else:
            print(msg)
        return kite
    except Exception:
        if interactive and st:
            st.warning("⚠️ Access token invalid or expired. Login required.")
            request_token = st.text_input(
                "Paste your request_token after logging in:",
                help="Visit https://kite.trade/connect/login?v=3&api_key={}".format(api_key)
            )
            if not request_token:
                st.stop()
        else:
            print("⚠️ Access token invalid or expired.")
            print("👉 Please login and get request_token here:")
            print(f"https://kite.trade/connect/login?v=3&api_key={api_key}")
            request_token = input("🔑 Paste request_token: ").strip()

        # Generate session
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        with open(token_path, "w") as f:
            json.dump({"access_token": access_token}, f)
        kite.set_access_token(access_token)
        profile = kite.profile()
        msg = f"✅ New access token saved. Logged in as {profile['user_name']}."
        if interactive and st:
            st.success(msg)
        else:
            print(msg)
        return kite
