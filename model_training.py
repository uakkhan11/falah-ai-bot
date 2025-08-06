# model_training.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import pandas_ta as ta

# ======================
# Step 1: Load historical data
# ======================
df = pd.read_csv("your_training_data.csv")
df.columns = [c.lower() for c in df.columns]
print("CSV Columns:", df.columns.tolist())

# ======================
# Step 2: Calculate Technical Indicators (if missing)
# ======================
if "rsi" not in df.columns:
    df["rsi"] = ta.rsi(df["close"], length=14)
if "ema10" not in df.columns:
    df["ema10"] = ta.ema(df["close"], length=10)
if "ema21" not in df.columns:
    df["ema21"] = ta.ema(df["close"], length=21)
if "atr" not in df.columns:
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
if "adx" not in df.columns:
    df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
if "volumechange" not in df.columns:
    df["volumechange"] = df["volume"].pct_change().fillna(0)

# ======================
# Step 3: Define Outcome
# ======================
df['future_high'] = df['close'].rolling(window=10, min_periods=1).max().shift(-1)
df['outcome'] = (df['future_high'] >= df['close'] * 1.05).astype(int)

# ======================
# Step 4: Feature List
# ======================
features = ["rsi", "atr", "adx", "ema10", "ema21", "volumechange"]

# Remove missing data
df = df.dropna(subset=features + ["outcome"])

# ======================
# Step 5: Filter last 2 years
# ======================
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
cutoff = pd.to_datetime("today").tz_localize(None) - pd.Timedelta(days=730)
df_recent = df[df["date"] >= cutoff]

# ======================
# Step 6: Prepare Inputs
# ======================
X = df_recent[features]
y = df_recent["outcome"]

print(f"✅ Dataset ready: {X.shape[0]} samples | Positives: {y.sum()} | Negatives: {(y==0).sum()}")

# ======================
# Step 7: Hyperparameter tuning (FAST - 20% sample)
# ======================
df_sample = df_recent.sample(frac=0.2, random_state=42)
X_sample = df_sample[features]
y_sample = df_sample["outcome"]

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

print("🔍 Running GridSearchCV on 20% sample...")
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_sample, y_sample)
best_params = grid_search.best_params_
print(f"✅ Best Params: {best_params}")

# ======================
# Step 8: Train FINAL model on FULL data
# ======================
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X, y)

# Cross-validation accuracy
scores = cross_val_score(final_model, X, y, cv=5)
print(f"✅ Cross-Validation Accuracy: {scores.mean():.4f}")

# Feature Importance
importances = final_model.feature_importances_
print("✅ Feature Importance:")
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.4f}")

# Save model
joblib.dump(final_model, "model.pkl")
print("✅ Final model saved as model.pkl")
