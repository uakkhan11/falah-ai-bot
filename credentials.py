# credentials.py
import json
from kiteconnect import KiteConnect

def load_secrets():
    """
    Load your credentials from the secrets JSON or TOML.
    """
    with open("/root/falah-ai-bot/secrets.json", "r") as f:
        secrets = json.load(f)
    return secrets

def get_kite():
    secrets = load_secrets()
    kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])
    if not os.path.exists("/root/falah-ai-bot/access_token.json"):
        raise Exception("❌ Access token file not found. Please generate it from dashboard UI.")
    with open("/root/falah-ai-bot/access_token.json", "r") as f:
        token = json.load(f)["access_token"]
    kite.set_access_token(token)
    return kite

def validate_kite(kite):
    """
    Check if access token is still valid by calling profile.
    """
    try:
        profile = kite.profile()
        print("✅ Valid credentials loaded.")
        return True
    except Exception as e:
        print(f"❌ Invalid credentials: {e}")
        return False
