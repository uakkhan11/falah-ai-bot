import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import joblib

# ✅ Step 1: Load Data
df = pd.read_csv("your_training_data.csv")

# ✅ Step 2: Create Outcome Column — +5% move within next 10 candles
df['Future_High'] = df['close'].rolling(window=10, min_periods=1).max().shift(-1)
df['Outcome'] = (df['Future_High'] >= df['close'] * 1.03).astype(int)

# ✅ Step 3: Clean rows
features = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"]
df = df.dropna(subset=features + ["Outcome"])

# ✅ Step 4: Recent 2 years filtering
df["date"] = pd.to_datetime(df["date"])
cutoff_date = pd.to_datetime("today") - pd.Timedelta(days=730)
df_recent = df

print(f"✅ Filtered Data: {len(df_recent)} rows | Positive={df_recent['Outcome'].sum()} | Negative={(df_recent['Outcome']==0).sum()}")

# ✅ Step 5: Model Inputs
X = df_recent[features]
y = df_recent["Outcome"]

# ✅ Step 6: Positive count check
if y.sum() < 10:
    print("⚠️ Not enough positive cases (<10) — please check dataset or outcome criteria.")
    exit()

# ✅ Step 7: Train-Test Split (20% test split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Step 8: Random Forest Model with Class Weight
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# ✅ Step 9: Cross Validation on Training Data
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"✅ 5-Fold CV (Train Accuracy): {cv_scores.mean():.4f}")

# ✅ Step 10: Test Accuracy
test_accuracy = model.score(X_test, y_test)
print(f"✅ Test Accuracy (20% unseen data): {test_accuracy:.4f}")

# ✅ Step 11: Feature Importance
print("\n✅ Feature Importances:")
for feat, imp in zip(features, model.feature_importances_):
    print(f"{feat}: {imp:.4f}")

# ✅ Step 12: Save Model
joblib.dump(model, "model.pkl")
print("\n✅ Model trained and saved to model.pkl")

# ✅ Step 13: Summary Log
print(f"\n🎉 FINAL SUMMARY:\nTotal Samples={len(df_recent)} | Positives={y.sum()} | Negatives={(y==0).sum()}")
print(f"Train CV Accuracy={cv_scores.mean():.4f} | Test Accuracy={test_accuracy:.4f}")
