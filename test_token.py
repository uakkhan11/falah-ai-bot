from kiteconnect import KiteConnect

kite = KiteConnect(api_key="ijzeuwuylr3g0kug")
kite.set_access_token("RxYIrGBDwFBclPqTfNy0WFqKs9OJ6V8a")

profile = kite.profile()
print("✅ Logged in as:", profile["user_name"])
