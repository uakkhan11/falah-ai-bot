# model_training_auto.py

import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# =========================
# Step 1: Load raw OHLCV data
# =========================
df = pd.read_csv("your_training_data.csv")  # Needs columns: date, open, high, low, close, volume
df["date"] = pd.to_datetime(df["date"])

# =========================
# Step 2: Calculate Indicators
# =========================
df["RSI"] = ta.rsi(df["close"], length=14)
df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
df["ADX"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
df["EMA10"] = ta.ema(df["close"], length=10)
df["EMA21"] = ta.ema(df["close"], length=21)
df["VolumeChange"] = df["volume"].pct_change().fillna(0)
df["MACD"] = ta.macd(df["close"])["MACDh_12_26_9"]
df["Stochastic"] = ta.stoch(df["high"], df["low"], df["close"])["STOCHK_14_3"]

# =========================
# Step 3: Create Outcome Variable (Next 10-day +5%)
# =========================
df['Future_High'] = df['close'].rolling(window=10, min_periods=1).max().shift(-1)
df['Outcome'] = (df['Future_High'] >= df['close'] * 1.05).astype(int)

# =========================
# Step 4: Filter recent 2 years & clean data
# =========================
cutoff = pd.to_datetime("today") - pd.Timedelta(days=730)
df = df[df["date"] >= cutoff]

features = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange", "MACD", "Stochastic"]
df = df.dropna(subset=features + ["Outcome"])

# =========================
# Step 5: Prepare Inputs
# =========================
X = df[features]
y = df["Outcome"]

print(f"✅ Dataset ready: {X.shape[0]} samples | Positive cases: {y.sum()} | Negative cases: {(y==0).sum()}")

# =========================
# Step 6: Hyperparameter Tuning
# =========================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_

# =========================
# Step 7: Save Models
# =========================
joblib.dump(best_model, "best_model.pkl")
joblib.dump(best_model, "model.pkl")  # For bot compatibility

print("✅ Best model saved as best_model.pkl & model.pkl")

# =========================
# Step 8: Feature Importance
# =========================
importances = best_model.feature_importances_
print("✅ Feature Importance:")
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.4f}")
