import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import pandas_ta as ta

# === CONFIG ===
CSV_PATH = "/root/falah-ai-bot/training_data.csv"
MODEL_PATH = "/root/falah-ai-bot/model.pkl"

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)
print(f"CSV Columns: {df.columns.tolist()}")

# Ensure date parsing
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
else:
    print("⚠️  'date' column not found — skipping date processing.")

# Sort by date
if "date" in df.columns:
    df.sort_values("date", inplace=True)

# Ensure numeric
numeric_cols = ["RSI", "ATR", "ADX"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# === FILL REQUIRED PRICE COLUMNS IF MISSING ===
if "close" not in df.columns:
    df["close"] = df["RSI"]  # placeholder for missing close
if "high" not in df.columns:
    df["high"] = df["close"] * 1.01
if "low" not in df.columns:
    df["low"] = df["close"] * 0.99
if "volume" not in df.columns:
    df["volume"] = 100000  # dummy volume

# === INDICATORS ===
# MACD
macd = ta.macd(df["close"])
df["macd_hist"] = macd["MACDh_12_26_9"]
df["macd_signal"] = macd["MACDs_12_26_9"]

# Stochastic Oscillator
stoch = ta.stoch(df["high"], df["low"], df["close"])
df["stoch_k"] = stoch["STOCHk_14_3_3"]
df["stoch_d"] = stoch["STOCHd_14_3_3"]

# OBV (On-Balance Volume)
df["obv"] = ta.obv(df["close"], df["volume"])

# VWAP
if "date" not in df.columns:
    print("⚠️  No 'date' column found — inserting dummy datetime index for VWAP.")
    df["date"] = pd.date_range(end=datetime.today(), periods=len(df), freq="1min")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df.dropna(subset=["date"], inplace=True)
df.set_index("date", inplace=True)

try:
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
except Exception as e:
    print(f"❌ VWAP computation failed: {e}")
    df["vwap"] = np.nan

df.reset_index(inplace=True)

# EMA10 & EMA21
df["ema10"] = ta.ema(df["close"], length=10)
df["ema21"] = ta.ema(df["close"], length=21)

# Volume Change
df["volumechange"] = df["volume"].pct_change().fillna(0)

# === CLEANUP ===
df.dropna(inplace=True)

# Target creation if missing
if "outcome" not in df.columns:
    if "Target" in df.columns:
        df["outcome"] = df["Target"].astype(int)
    else:
        raise ValueError("Target or outcome column not found.")

# Final features and target
features = [
    "RSI", "ATR", "ADX", "ema10", "ema21", "volumechange",
    "macd_hist", "macd_signal", "stoch_k", "stoch_d", "obv", "vwap"
]
target = "outcome"

X = df[features]
y = df[target]

print(f"✅ Dataset ready: {len(X)} samples | Positives: {sum(y)} | Negatives: {len(y) - sum(y)}")

# === MODEL TRAINING PIPELINE ===
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42))
])

param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 5],
    "clf__class_weight": ["balanced"]
}

X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.8, random_state=42, stratify=y)

grid = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
print("🔍 Running GridSearchCV on 20% sample...")
grid.fit(X_sample, y_sample)

print(f"✅  Best Params: {grid.best_params_}")

# === CROSS VALIDATION ===
scores = cross_val_score(grid.best_estimator_, X, y, cv=5, scoring="accuracy")
print(f"✅ Cross-Validation Accuracy: {scores.mean():.4f}")

# === FEATURE IMPORTANCE ===
best_model = grid.best_estimator_.named_steps["clf"]
importances = best_model.feature_importances_
feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

print("✅ Feature Importance:")
for f, imp in feat_imp:
    print(f"{f}: {imp:.4f}")

# === SAVE MODEL ===
joblib.dump(grid.best_estimator_, MODEL_PATH)
print(f"✅  Final model saved as {MODEL_PATH}")
