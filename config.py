# config.py
import toml
import os

def load_secrets():
    # Load from the .streamlit/secrets.toml file
    secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
    with open(secrets_path, "r") as f:
        secrets = toml.load(f)
    return secrets
