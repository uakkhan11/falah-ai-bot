import json
from kiteconnect import KiteConnect

# Your credentials
api_key = "ijzeuwuylr3g0kug"
api_secret = "yy1wd2wn8r0wx4mus00vxllgss03nuqx"

# IMPORTANT:
# You must manually get this request token after logging in
request_token = "PASTE_YOUR_REQUEST_TOKEN_HERE"

# Create KiteConnect instance
kite = KiteConnect(api_key=api_key)

# Generate session
data = kite.generate_session(
    request_token=request_token,
    api_secret=api_secret
)

access_token = data["access_token"]
kite.set_access_token(access_token)

print("✅ Access token generated:", access_token)

# Save access token to file
with open("/root/falah-ai-bot/access_token.json", "w") as f:
    json.dump({"access_token": access_token}, f)

print("✅ Access token saved to /root/falah-ai-bot/access_token.json")
