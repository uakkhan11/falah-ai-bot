import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os

# ---- Step 1: Load CSV ----
filename = "training_data.csv"  # or change to your CSV filename
if not os.path.exists(filename):
    raise FileNotFoundError(f"❌ File '{filename}' not found!")

df = pd.read_csv(filename)
print("✅ CSV Columns:", list(df.columns))

# ---- Step 2: Optional Date Handling ----
if 'date' in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
else:
    print("⚠️  'date' column not found — skipping date processing.")

# ---- Step 3: Select Features and Target ----
expected_features = ["RSI", "ATR", "ADX"]
missing = [col for col in expected_features if col not in df.columns]
if missing:
    raise ValueError(f"❌ Missing expected columns: {missing}")

X = df[expected_features]
y = df["Target"]

# ---- Step 4: Train/Test Split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Step 5: Train the Model ----
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- Step 6: Evaluate Performance ----
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# ---- Step 7: Save Model ----
joblib.dump(model, "model.pkl")
print("💾 Model saved to 'model.pkl'")
