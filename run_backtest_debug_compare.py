import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower() for c in df.columns]

required_cols = ['date', 'close', 'high', 'low', 'volume', 'rsi', 'atr', 'adx', 'target', 'outcome']
for col in ['close', 'high', 'low', 'volume']:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in dataset")

# Parse dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date'], inplace=True)
df.sort_values('date', inplace=True)

# Feature engineering
df['macd'] = ta.macd(df['close'])['MACDh_12_26_9']
df['macd_signal'] = ta.macd(df['close'])['MACDs_12_26_9']
df['stoch_k'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHk_14_3_3']
df['stoch_d'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHd_14_3_3']
df['obv'] = ta.obv(df['close'], df['volume'])
df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
df['ema10'] = ta.ema(df['close'], length=10)
df['ema21'] = ta.ema(df['close'], length=21)
df['volumechange'] = df['volume'].pct_change().fillna(0)

# Drop rows with na in features or target
features = ["rsi", "atr", "adx", "ema10", "ema21", "volumechange",
            "macd", "macd_signal", "stoch_k", "stoch_d", "obv", "vwap"]

target = 'outcome'  # ensure your data uses this target column

df.dropna(subset=features + [target], inplace=True)

X = df[features]
y = df[target].astype(int)

# Train/test split 20% for grid search to save time
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5],
    'clf__class_weight': ['balanced']
}

grid = GridSearchCV(pipe, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_sample, y_sample)

print(f"Best params on sample: {grid.best_params_}")

# Evaluate full data with best model
best_model = grid.best_estimator_
scores = cross_val_score(best_model, X, y, cv=5, scoring='f1')
print(f"Cross-validation F1 score: {scores.mean():.4f}")

y_pred = best_model.predict(X)
print(classification_report(y, y_pred))

# Feature importances
rf = best_model.named_steps['clf']
importances = rf.feature_importances_
for f, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    print(f"{f}: {imp:.4f}")

# Save model
joblib.dump(best_model, MODEL_PATH)
