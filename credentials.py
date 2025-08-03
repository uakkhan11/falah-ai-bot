# credentials.py

import json

def load_secrets(path="secrets.json"):
    with open(path, "r") as f:
        return json.load(f)

def get_kite_tokens():
    secrets = load_secrets()
    return secrets["api_key"], secrets["access_token"]
