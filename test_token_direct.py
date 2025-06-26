from kiteconnect import KiteConnect

api_key = "ijzeuwuylr3g0kug"
access_token = "1r85UUlnvKFh6wRRSSnkiEqjyBL0cJke"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

try:
    profile = kite.profile()
    print("✅ VALID TOKEN")
    print("Logged in as:", profile["user_name"])
except Exception as e:
    print("❌ INVALID TOKEN")
    print(e)
