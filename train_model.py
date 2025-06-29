import pandas as pd
import os
import glob
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

DATA_FOLDER = "/root/falah-ai-bot/historical_data"

csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
all_features = []
print(f"Found {len(csv_files)} CSV files.")

for file in csv_files:
    symbol = os.path.basename(file).replace(".csv", "")
    try:
        df = pd.read_csv(file)
        df["symbol"] = symbol

        df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
        df["ema_10"] = EMAIndicator(df["close"], window=10).ema_indicator()
        df["ema_50"] = EMAIndicator(df["close"], window=50).ema_indicator()
        df["macd"] = MACD(df["close"]).macd_diff()

        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.dropna()
        all_features.append(df)
    except Exception as e:
        print(f"❌ Skipping {symbol}: {e}")

data = pd.concat(all_features, ignore_index=True)
print("✅ Data shape:", data.shape)

X = data[["rsi", "ema_10", "ema_50", "macd"]]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

with open("/root/falah-ai-bot/model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Model saved as model.pkl")
