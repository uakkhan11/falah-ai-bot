from kiteconnect import KiteConnect

api_key = "ijzeuwuylr3g0kug"
api_secret = "yy1wd2wn8r0wx4mus00vxllgss03nuqx"
request_token = "b1G8wI6pd44GWXZTS384nwg5KYtnKuo6"   # from URL

kite = KiteConnect(api_key=api_key)
data = kite.generate_session(request_token, api_secret=api_secret)
print("✅ ACCESS TOKEN:", data["access_token"])
