# auto_auth.py

import json
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import webbrowser
from kiteconnect import KiteConnect

TOKENS_FILE = "kite_tokens.json"
API_KEY = "your_api_key_here"
API_SECRET = "your_api_secret_here"
REDIRECT_PORT = 8080
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}"  # Must match your app setting

class TokenHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if "request_token" in params:
            self.server.request_token = params["request_token"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h2>Token received! You can close this window.</h2>")
        else:
            self.send_response(400)
            self.end_headers()
    def log_message(self, format, *args):
        pass  # Silence default logging

def auto_authenticate():
    # 1. Try loading a saved access token
    if os.path.exists(TOKENS_FILE):
        try:
            with open(TOKENS_FILE) as f:
                data = json.load(f)
                token = data.get("access_token")
                if token:
                    kite = KiteConnect(api_key=API_KEY)
                    kite.set_access_token(token)
                    try:
                        profile = kite.profile()
                        print("✅ Authenticated with saved token:", profile["user_name"])
                        return kite
                    except:
                        print("⚠️ Saved token expired, obtaining new token.")
        except Exception as e:
            print("⚠️ Could not load saved token:", e)

    # 2. Start a local HTTP server to catch the request_token
    server = HTTPServer(("localhost", REDIRECT_PORT), TokenHandler)
    server.request_token = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # 3. Open KiteConnect login URL in your default browser
    kite = KiteConnect(api_key=API_KEY)
    login_url = kite.login_url()
    print("🔑 Opening login URL in your browser...")
    webbrowser.open(login_url)

    # 4. Wait for the request_token to arrive
    timeout = 60  # seconds
    print(f"⏳ Waiting up to {timeout}s for login & redirect...")
    while timeout and not server.request_token:
        time.sleep(1)
        timeout -= 1
    server.shutdown()

    if not server.request_token:
        print("❌ Timeout waiting for request_token.")
        return None

    # 5. Exchange the request_token for an access_token
    try:
        session = kite.generate_session(server.request_token, api_secret=API_SECRET)
        access_token = session["access_token"]
        kite.set_access_token(access_token)
        # 6. Save the access_token for future runs
        with open(TOKENS_FILE, "w") as f:
            json.dump({"access_token": access_token}, f)
        print("✅ Authentication successful. Token saved.")
        return kite
    except Exception as e:
        print("❌ Token exchange failed:", e)
        return None

if __name__ == "__main__":
    kite = auto_authenticate()
    if kite:
        print("🎉 Ready to trade!")
