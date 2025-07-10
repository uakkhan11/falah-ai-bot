# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
df = pd.read_csv("training_data.csv")

# Use available columns
X = df[["RSI", "EMA10", "EMA21", "ATR", "VolumeChange"]]
y = df["Target"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "/root/falah-ai-bot/model.pkl")
print("✅ Model saved as model.pkl")

