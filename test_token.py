from kiteconnect import KiteConnect

kite = KiteConnect(api_key="ijzeuwuylr3g0kug")
kite.set_access_token("eukSPf050njsE6ucsN9RbIg5RbSpaosh")

profile = kite.profile()
print("✅ Logged in as:", profile["user_name"])
