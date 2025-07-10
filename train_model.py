# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

INPUT_FILE = "/root/falah-ai-bot/training_data.csv"
MODEL_FILE = "/root/falah-ai-bot/model.pkl"

df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df)} training rows.")

X = df[["RSI", "ATR", "ADX", "AI_Score"]]
y = df["Target"]

model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X, y)

joblib.dump(model, MODEL_FILE)
print(f"✅ Model trained and saved to {MODEL_FILE}.")
