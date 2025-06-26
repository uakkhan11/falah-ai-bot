from kiteconnect import KiteConnect

kite = KiteConnect(api_key="ijzeuwuylr3g0kug")
kite.set_access_token("053g791myYRvFIjV49pFsFz51W72k0gT")

profile = kite.profile()
print("✅ Logged in as:", profile["user_name"])
