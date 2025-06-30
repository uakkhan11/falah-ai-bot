import pyotp
from kiteconnect import KiteConnect

# Your credentials
api_key = "ijzeuwuylr3g0kug"
api_secret = "yy1wd2wn8r0wx4mus00vxllgss03nuqx"
user_id = "RVM407"
totp_secret = "3Z6DA326HX5ZXT6VPZDT3AY63TUV233T"  # your TOTP secret

# Create KiteConnect
kite = KiteConnect(api_key=api_key)

# Generate TOTP dynamically
totp = pyotp.TOTP(totp_secret)
otp = totp.now()

# Request session
request_token = kite.generate_session(
    user_id=user_id,
    password="your_password",
    twofa=otp
)
access_token = request_token["access_token"]

# Set token
kite.set_access_token(access_token)

# Test
profile = kite.profile()
print("✅ Logged in:", profile["user_name"])

# Your existing login logic:
from kiteconnect import KiteConnect

kite = KiteConnect(api_key="...")
data = kite.generate_session(
    request_token=your_request_token,
    api_secret="..."
)
access_token = data["access_token"]

# Save access token to JSON file:
with open("/root/falah-ai-bot/access_token.json", "w") as f:
    import json
    json.dump({"access_token": access_token}, f)

print("✅ Access token saved to access_token.json")

