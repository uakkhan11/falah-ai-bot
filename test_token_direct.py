from kiteconnect import KiteConnect

api_key = "PUT_API_KEY_HERE"
access_token = "PUT_ACCESS_TOKEN_HERE"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

try:
    profile = kite.profile()
    print("✅ VALID TOKEN")
    print("Logged in as:", profile["user_name"])
except Exception as e:
    print("❌ INVALID TOKEN")
    print(e)
