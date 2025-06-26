from kiteconnect import KiteConnect

kite = KiteConnect(api_key="ijzeuwuylr3g0kug")
kite.set_access_token("5xRl14oo95VnhV5iClVi1A26BXfC2wBW")  # paste the exact token printed by generate_session()

profile = kite.profile()
print("✅ Profile loaded:", profile["user_name"])
