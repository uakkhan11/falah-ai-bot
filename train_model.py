# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("training_data.csv")
X = df[["RSI", "ATR"]]   # Only columns you have
y = df["Target"]

model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved with 2 features: RSI, ATR")
