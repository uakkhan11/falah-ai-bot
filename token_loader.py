# token_loader.py

import json
from kiteconnect import KiteConnect

def load_or_create_token(api_key, api_secret, interactive=True, st=None):
    kite = KiteConnect(api_key=api_key)
    
    # Check if access_token.json exists
    try:
        with open("/root/falah-ai-bot/access_token.json", "r") as f:
            data = json.load(f)
            access_token = data["access_token"]
            kite.set_access_token(access_token)
            # Test if token works
            kite.profile()
            if interactive and st:
                st.success("✅ Valid access token loaded.")
            return kite
    except Exception as e:
        if interactive and st:
            st.warning("⚠️ Access token missing or invalid.")
        else:
            print("⚠️ Access token missing or invalid.")
    
    # Show login URL
    login_url = f"https://kite.trade/connect/login?v=3&api_key={api_key}"
    if interactive and st:
        st.markdown(f"""
            ### 🔑 Login Required
            [Click here to login to Zerodha and get your request token]({login_url})
            """)
        request_token = st.text_input("Paste your request_token here:")
        
        if request_token:
            try:
                data = kite.generate_session(request_token, api_secret)
                access_token = data["access_token"]
                kite.set_access_token(access_token)
                with open("/root/falah-ai-bot/access_token.json", "w") as f:
                    json.dump({"access_token": access_token}, f)
                st.success("✅ Access token generated and saved.")
                return kite
            except Exception as e:
                st.error(f"❌ Failed to generate token: {e}")
                st.stop()
        else:
            st.stop()
    else:
        print(f"🔗 Login URL: {login_url}")
        request_token = input("Paste your request_token: ").strip()
        data = kite.generate_session(request_token, api_secret)
        access_token = data["access_token"]
        kite.set_access_token(access_token)
        with open("/root/falah-ai-bot/access_token.json", "w") as f:
            json.dump({"access_token": access_token}, f)
        print("✅ Access token generated and saved.")
        return kite
