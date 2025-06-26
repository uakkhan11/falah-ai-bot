from kiteconnect import KiteConnect
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pyotp, time, toml, tempfile

# Load secrets
secrets = toml.load("/root/falah-ai-bot/.streamlit/secrets.toml")
api_key = secrets["zerodha"]["api_key"]
api_secret = secrets["zerodha"]["api_secret"]
totp_secret = secrets["zerodha"]["totp"]
user_id = secrets["zerodha"]["user_id"]
password = secrets["zerodha"]["password"]

kite = KiteConnect(api_key=api_key)
totp = pyotp.TOTP(totp_secret).now()

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# ✅ Fix: Remove --user-data-dir to avoid lock issue
# tmp_profile = tempfile.mkdtemp()
# options.add_argument(f"--user-data-dir={tmp_profile}")

driver = None  # ✅ Prevent NameError in finally block

try:
    driver = webdriver.Chrome(options=options)  # Chrome starts here
    login_url = kite.login_url()
    driver.get(login_url)

    driver.find_element(By.ID, "userid").send_keys(user_id)
    driver.find_element(By.ID, "password").send_keys(password)
    driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]').click()

    time.sleep(2)
    driver.find_element(By.ID, "pin").send_keys(totp)
    driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]').click()

    time.sleep(2)
    current_url = driver.current_url
    if "request_token=" not in current_url:
        raise Exception("❌ Failed to retrieve request_token.")
    
    request_token = current_url.split("request_token=")[1].split("&")[0]
    print("✅ request_token:", request_token)

except Exception as e:
    print(f"❌ Error during login flow: {e}")

finally:
    if driver:
        driver.quit()

# ✅ Generate access token from request token
data = kite.generate_session(request_token, api_secret=api_secret)
access_token = data["access_token"]
print("✅ access_token:", access_token)

# ✅ Save to secrets
secrets["zerodha"]["access_token"] = access_token
with open("/root/falah-ai-bot/.streamlit/secrets.toml", "w") as f:
    toml.dump(secrets, f)

print("✅ Updated access_token in secrets.toml successfully.")
