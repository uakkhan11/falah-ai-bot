# model_training.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import joblib

# ✅ Step 1: Load Data
df = pd.read_csv("your_training_data.csv")

# ✅ Step 2: Clean Data
features = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"]
df = df.dropna(subset=features + ["Outcome"])

if len(df) > 100000:
    df = df.sample(100000, random_state=42)


print(f"✅ Data Loaded: {len(df)} rows | Positive={df['Outcome'].sum()} | Negative={(df['Outcome']==0).sum()}")

# ✅ Step 3: Prepare Inputs
X = df[features]
y = df["Outcome"]

# ✅ Step 4: Positive count check
if y.sum() < 10:
    print("⚠️ Not enough positive cases (<10). Please check dataset or outcome criteria.")
    exit()

# ✅ Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Step 6: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# ✅ Step 7: Cross Validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"✅ Cross-Validation Accuracy (Train): {cv_scores.mean():.4f}")

# ✅ Step 8: Test Accuracy
test_accuracy = model.score(X_test, y_test)
print(f"✅ Test Accuracy (20% unseen data): {test_accuracy:.4f}")

# ✅ Step 9: Feature Importance
print("\n✅ Feature Importances:")
for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# ✅ Step 10: Save Model
joblib.dump(model, "model.pkl")
print("\n✅ Model trained and saved to model.pkl")
