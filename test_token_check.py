# test_token_check.py
from kiteconnect import KiteConnect

api_key = "ijzeuwuylr3g0kug"
access_token = "IhoPB4YfvtPxmgn7MJBziBwG3QCIo4La"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

try:
    profile = kite.profile()
    print("✅ Logged in as:", profile["user_name"])
except Exception as e:
    print("❌ Token failed:", str(e))
