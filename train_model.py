import pandas as pd
import os
import glob
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Folder where your CSVs are stored
DATA_FOLDER = "/root/falah-ai-bot/data"

# Gather all CSV files
csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

all_features = []

print(f"Found {len(csv_files)} CSV files.")

for file in csv_files:
    symbol = os.path.basename(file).replace(".csv", "")
    try:
        df = pd.read_csv(file)
        df["symbol"] = symbol
        
        # Technical Indicators
        df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
        df["ema_10"] = EMAIndicator(df["close"], window=10).ema_indicator()
        df["ema_50"] = EMAIndicator(df["close"], window=50).ema_indicator()
        macd = MACD(df["close"])
        df["macd"] = macd.macd_diff()
        
        # Target variable: will next day's close be higher?
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        
        # Drop last row (no next day)
        df = df[:-1]
        
        all_features.append(df)
    except Exception as e:
        print(f"❌ Skipping {symbol}: {e}")

# Combine all into one DataFrame
data = pd.concat(all_features, ignore_index=True)
print("✅ Data shape:", data.shape)

# Drop rows with NaNs (from indicators)
data = data.dropna()

# Features and target
X = data[["rsi", "ema_10", "ema_50", "macd"]]
y = data["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Save model
with open("/root/falah-ai-bot/model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Model saved as model.pkl")
