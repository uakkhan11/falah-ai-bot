import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("training_data.csv")

# Remove incomplete rows
df = df.dropna(subset=["RSI", "EMA10", "EMA21", "ATR", "VolumeChange", "Target"])

# Define inputs and target
features = ["RSI", "EMA10", "EMA21", "ATR", "VolumeChange"]
X = df[features]
y = df["Outcome"]

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=6  # optional: limit tree depth for generalization
)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "/root/falah-ai-bot/model.pkl")
print("✅ Model trained and saved successfully.")

# Show feature names
print("✅ Model trained with features:", model.feature_names_in_)
