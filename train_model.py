import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("training_data.csv")

features = ["RSI", "EMA10", "EMA21", "SMA20", "ATR", "VolumeChange"]
X = df[features]
y = df["Target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
