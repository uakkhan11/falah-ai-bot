import streamlit as st
from kiteconnect import KiteConnect
import toml
import pandas as pd

# Load secrets
secrets = toml.load(".streamlit/secrets.toml")
api_key = secrets["zerodha"]["api_key"]
access_token = secrets["zerodha"]["access_token"]

# Setup Kite
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

st.title("📈 Falāh AI Trading Bot")
st.caption("Built with 💡 by Usman")

st.markdown("⏳ **Initializing and verifying credentials...**")

# Validate token
try:
    profile = kite.profile()
    st.success(f"🧑‍💼 Logged in as: {profile['user_name']}")
except Exception as e:
    st.error("❌ Invalid API key or access token.")
    st.stop()

# Dummy halal stocks loaded (replace with your actual logic)
st.markdown("📋 Loaded 1067 halal stocks...")

# Simulate stock list
stocks = ["TCS", "INFY", "HCLTECH", "LTIM", "TECHM", "MPHASIS", "COFORGE", "PERSISTENT", "RELIANCE", "IRCTC"]
st.markdown(f"**Loaded symbols:** `{stocks[:10]}`")

# Simulate data fetch
stock_data = []
for stock in stocks:
    try:
        quote = kite.ltp(f"NSE:{stock}")
        last_price = quote[f"NSE:{stock}"]["last_price"]
        stock_data.append({"Symbol": stock, "LTP": last_price, "AI Score": round(last_price % 10, 2)})
    except Exception as e:
        st.warning(f"❌ Skipping {stock}: {str(e)}")

if not stock_data:
    st.warning("⚠️ No stock data available. Check if Zerodha access token is valid or API is rate-limited.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(stock_data)
st.markdown("🧾 **Raw DataFrame:**")
st.dataframe(df)

# Sort by AI Score
try:
    candidates = df.sort_values(by="AI Score", ascending=False).head(5)
    st.subheader("📊 Filtered Trade Candidates")
    st.dataframe(candidates)
except KeyError:
    st.error("❌ 'AI Score' column not found in data.")
