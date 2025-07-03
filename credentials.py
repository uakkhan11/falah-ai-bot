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
    """
    Returns an initialized KiteConnect object with access token set.
    """
    secrets = load_secrets()
    api_key = secrets["zerodha"]["api_key"]

    with open("/root/falah-ai-bot/access_token.json", "r") as f:
        access_token = json.load(f)["access_token"]

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
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
