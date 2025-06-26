import toml
from utils import load_credentials

creds = load_credentials()
print("✅ Loaded credentials:")
print(creds)

try:
    secrets = toml.load("/root/falah-ai-bot/.streamlit/secrets.toml")
    print("✅ Secrets loaded:")
    print(secrets)
except Exception as e:
    print("❌ Error loading secrets:", e)
