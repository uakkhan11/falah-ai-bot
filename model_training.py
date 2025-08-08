import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
import joblib
import warnings

warnings.filterwarnings("ignore")  # Optional: suppress warnings for cleaner output

# ======================
# Configurations
# ======================
CSV_PATH = "your_training_data.csv"
MODEL_SAVE_PATH = "model.pkl"
TARGET_COLUMN = "outcome"  # Lowercase assumed

# ======================
# Step 1: Load historical data
# ======================
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower() for c in df.columns]  # Standardize column names
print("CSV Columns:", df.columns.tolist())

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df["date"] = df["date"].dt.tz_localize(None)  # Remove timezone info
    df.sort_values("date", inplace=True)
else:
    print("⚠️ 'date' column not found - date filtering will be skipped")

# ======================
# Step 2: Calculate Technical Indicators (if missing)
# ======================
required_price_cols = ["close"]
if any(col not in df.columns for col in required_price_cols):
    raise ValueError(f"Required columns missing from dataset: {required_price_cols}")

if "rsi" not in df.columns:
    df["rsi"] = ta.rsi(df["close"], length=14)

if all(col in df.columns for col in ["high", "low", "close"]):
    if "atr" not in df.columns:
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    if "adx" not in df.columns:
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx_df["adx_14"]
else:
    print("⚠️ Missing 'high' or 'low' columns - skipping 'atr' and 'adx' calculations")

if "ema10" not in df.columns:
    df["ema10"] = ta.ema(df["close"], length=10)
if "ema21" not in df.columns:
    df["ema21"] = ta.ema(df["close"], length=21)

if "volume" in df.columns:
    if "volumechange" not in df.columns:
        df["volumechange"] = df["volume"].pct_change().fillna(0)
else:
    print("⚠️ 'volume' column missing - 'volumechange' feature skipped")

# ======================
# Step 3: Define Outcome
# ======================
if TARGET_COLUMN not in df.columns:
    df["future_high"] = df["close"].rolling(window=10, min_periods=1).max().shift(-1)
    df[TARGET_COLUMN] = (df["future_high"] >= df["close"] * 1.05).astype(int)
else:
    print(f"Using existing target column: {TARGET_COLUMN}")

# ======================
# Step 4: Filter last 2 years if 'date' is present
# ======================
if "date" in df.columns:
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=730)
    df_recent = df[df["date"] >= cutoff].copy()
    print(f"Filtered last 2 years data: {df_recent.shape[0]} rows")
else:
    df_recent = df.copy()
    print("No date filtering applied")

# ======================
# Step 5: Define features, clean data, and prepare inputs
# ======================
features = ["rsi", "atr", "adx", "ema10", "ema21", "volumechange"]
features = [f for f in features if f in df_recent.columns]  # Ensure features exist

before_len = len(df_recent)
df_recent.dropna(subset=features + [TARGET_COLUMN], inplace=True)
after_len = len(df_recent)

print(f"Dropped {before_len - after_len} rows due to missing data")

X = df_recent[features]
y = df_recent[TARGET_COLUMN].astype(int)

print(f"✅ Dataset ready: {X.shape[0]} samples | Positives: {y.sum()} | Negatives: {len(y) - y.sum()}")

# ======================
# Step 6: Hyperparameter tuning (RandomizedSearch for faster tuning)
# ======================
X_sample, _, y_sample, _ = train_test_split(
    X, y, test_size=0.9, stratify=y, random_state=42
)  # Using 10% for tuning to speed up (adjust as needed)

param_dist = {
    "n_estimators": [100, 150],
    "max_depth": [None, 10],
    "min_samples_split": [2],
    "class_weight": ["balanced"],  # Important for imbalanced data
}

print("🔍 Running RandomizedSearchCV on 10% sample...")
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=5,  # Number of parameter settings sampled, adjust as desired
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
    random_state=42,
)
random_search.fit(X_sample, y_sample)
print(f"✅ Best Params: {random_search.best_params_}")

# ======================
# Step 7: Train final model on full dataset and evaluate
# ======================
final_model = RandomForestClassifier(
    **random_search.best_params_, random_state=42
)
final_model.fit(X, y)

cv_scores = cross_val_score(final_model, X, y, cv=5, scoring="f1")
print(f"✅ 5-Fold Cross-Validation F1 Score: {cv_scores.mean():.4f}")

y_pred = final_model.predict(X)
print("\nClassification report on training data:")
print(classification_report(y, y_pred))

# ======================
# Step 8: Feature Importances and model save
# ======================
importances = final_model.feature_importances_
print("✅ Feature Importances:")
for f, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    print(f"  {f}: {imp:.4f}")

joblib.dump(final_model, MODEL_SAVE_PATH)
print(f"✅ Final model saved as {MODEL_SAVE_PATH}")
