from kiteconnect import KiteConnect
import json

api_key = "ijzeuwuylr3g0kug"
api_secret = "yy1wd2wn8r0wx4mus00vxllgss03nuqx"
request_token = "ljM3esGSngie1jEJoPDoxiLqySG0co6h"   # from URL

kite = KiteConnect(api_key=api_key)
data = kite.generate_session(request_token, api_secret=api_secret)
print("✅ ACCESS TOKEN:", data["access_token"])

access_token = data["access_token"]
kite.set_access_token(access_token)

print("✅ Access token generated:", access_token)

# Save access token to file
with open("/root/falah-ai-bot/access_token.json", "w") as f:
    json.dump({"access_token": access_token}, f)

print("✅ Access token saved to /root/falah-ai-bot/access_token.json")
