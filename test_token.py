from kiteconnect import KiteConnect

kite = KiteConnect(api_key="ijzeuwuylr3g0kug")
kite.set_access_token("vD50h2d0h866YZaesuXxgSjl2Se3NaYF")

profile = kite.profile()
print("✅ Logged in as:", profile["user_name"])
