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

# === ENSURE REQUIRED COLUMNS ===
if "date" not in df.columns:
    print("⚠️  'date' column not found — inserting dummy datetimes.")
    df["date"] = pd.date_range(end=datetime.today(), periods=len(df), freq="1min")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df.dropna(subset=["date"], inplace=True)
df.set_index("date", inplace=True)

# Fill missing price/volume columns
for col, default in zip(["close", "high", "low", "volume"], [100, 101, 99, 100000]):
    if col not in df.columns:
        df[col] = default if col != "close" else df["RSI"]

# === CLEAN NUMERIC COLUMNS ===
numeric_cols = ["RSI", "ATR", "ADX", "close", "high", "low", "volume"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# === INDICATORS ===
indicators = {}
indicators["macd"] = ta.macd(df["close"])
indicators["stoch"] = ta.stoch(df["high"], df["low"], df["close"])
df["macd_hist"] = indicators["macd"]["MACDh_12_26_9"]
df["macd_signal"] = indicators["macd"]["MACDs_12_26_9"]
df["stoch_k"] = indicators["stoch"]["STOCHk_14_3_3"]
df["stoch_d"] = indicators["stoch"]["STOCHd_14_3_3"]
df["obv"] = ta.obv(df["close"], df["volume"])

try:
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
except Exception as e:
    print(f"❌ VWAP failed: {e}")
    df["vwap"] = np.nan

df["ema10"] = ta.ema(df["close"], length=10)
df["ema21"] = ta.ema(df["close"], length=21)
df["volumechange"] = df["volume"].pct_change().fillna(0)

df.dropna(inplace=True)
df.reset_index(inplace=True)

# === TARGET COLUMN ===
if "outcome" not in df.columns:
    if "Target" in df.columns:
        df["outcome"] = df["Target"].astype(int)
    else:
        raise ValueError("No outcome or Target column found.")

features = [
    "RSI", "ATR", "ADX", "ema10", "ema21", "volumechange",
    "macd_hist", "macd_signal", "stoch_k", "stoch_d", "obv", "vwap"
]
target = "outcome"
X = df[features]
y = df[target]

print(f"✅ Data Ready: {len(X)} samples | Positives: {sum(y)} | Negatives: {len(y) - sum(y)}")

# === MODEL PIPELINE & TRAINING ===
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42))
])

param_grid = {
    "clf__n_estimators": [100],
    "clf__max_depth": [None, 10],
    "clf__min_samples_split": [2],
    "clf__class_weight": ["balanced"]
}

X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.8, random_state=42, stratify=y)
print("🔍 GridSearchCV on 20% sample...")
grid = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid.fit(X_sample, y_sample)

print(f"✅ Best Params: {grid.best_params_}")
scores = cross_val_score(grid.best_estimator_, X, y, cv=5, scoring="accuracy")
print(f"✅ Cross-Validation Accuracy: {scores.mean():.4f}")

# === FEATURE IMPORTANCE ===
model = grid.best_estimator_.named_steps["clf"]
feat_imp = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
print("✅ Feature Importance:")
for f, imp in feat_imp:
    print(f"{f}: {imp:.4f}")

# === SAVE MODEL ===
joblib.dump(grid.best_estimator_, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
