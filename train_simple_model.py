import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load CSV
df = pd.read_csv("training_data.csv")
print("✅ CSV Columns:", df.columns.tolist())

# Date Handling
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
else:
    print("⚠️  'date' column not found — skipping date processing.")

# Required columns
required_cols = ["RSI", "ATR", "ADX", "Target"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing required columns: {missing_cols}")

# Convert to numeric
df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")
df.dropna(subset=required_cols, inplace=True)

# Feature + target split
X = df[["RSI", "ATR", "ADX"]]
y = df["Target"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict + evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Results
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print(f"✅ Accuracy: {acc:.2%}")
