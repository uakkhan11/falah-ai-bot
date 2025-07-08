from kiteconnect import KiteConnect

API_KEY = "ijzeuwuylr3g0kug"
ACCESS_TOKEN = "AmrLL5hewKt70qzmrJCisiJEmlSP9b4M"  # This may already be expired

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

try:
    profile = kite.profile()
    print("✅ API Key and Access Token are valid!")
except Exception as e:
    print(f"❌ Invalid credentials: {e}")
