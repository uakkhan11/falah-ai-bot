# model_training_auto.py

# training.py (Updated with MACD + Stochastic)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import pandas_ta as ta

# ✅ Step 1: Load historical data
df = pd.read_csv("your_training_data.csv")

# ✅ Step 2: Calculate Technical Indicators (to match live scanner)
df["RSI"] = ta.rsi(df["close"], length=14)
df["EMA10"] = ta.ema(df["close"], length=10)
df["EMA21"] = ta.ema(df["close"], length=21)
df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
df["MACD"] = ta.macd(df["close"])["MACDh_12_26_9"]
df["Stochastic"] = ta.stoch(df["high"], df["low"], df["close"])["STOCHK_14_3"]
df["VolumeChange"] = df["volume"].pct_change().fillna(0)

# ✅ Step 3: Calculate Outcome (Next 10-day +5% target)
df['Future_High'] = df['close'].rolling(window=10, min_periods=1).max().shift(-1)
df['Outcome'] = (df['Future_High'] >= df['close'] * 1.05).astype(int)

# ✅ Step 4: Feature Columns (now includes MACD & Stochastic)
features = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange", "MACD", "Stochastic"]

# Drop rows with missing data
df = df.dropna(subset=features + ["Outcome"])

# ✅ Step 5: Filter for recent 2 years
df["date"] = pd.to_datetime(df["date"])
cutoff = pd.to_datetime("today") - pd.Timedelta(days=730)
df_recent = df[df["date"] >= cutoff]

# ✅ Step 6: Prepare data
X = df_recent[features]
y = df_recent["Outcome"]

print(f"✅ Dataset ready: {X.shape[0]} samples | Positives: {y.sum()} | Negatives: {(y==0).sum()}")

# ✅ Step 7: Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

print("🔍 Running GridSearchCV for best RandomForest parameters...")
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

model = grid_search.best_estimator_
print(f"✅ Best Params: {grid_search.best_params_}")

# ✅ Step 8: Cross-validation score
scores = cross_val_score(model, X, y, cv=5)
print(f"✅ Cross-Validation Accuracy: {scores.mean():.4f}")

# ✅ Step 9: Feature importance
importances = model.feature_importances_
print("✅ Feature Importance:")
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.4f}")

# ✅ Step 10: Save model
joblib.dump(model, "model.pkl")
print("✅ Final model saved as model.pkl")
