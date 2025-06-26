from kiteconnect import KiteConnect

kite = KiteConnect(api_key="ijzeuwuylr3g0kug")
kite.set_access_token("8e1HmKugtXQB12ncAYEDsfIcdbett6hK")

profile = kite.profile()
print("✅ Logged in as:", profile["user_name"])
