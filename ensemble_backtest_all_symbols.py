
# progress_ensemble_backtest.py

import os
import pandas as pd
import numpy as np
import ta
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    'daily': os.path.join(BASE_DIR, "swing_data"),
}

def list_symbols():
    return [f[:-4] for f in os.listdir(DATA_PATHS['daily']) if f.endswith('.csv')]

def load_data(symbol, source):
    path = os.path.join(DATA_PATHS[source], f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year == 2025].reset_index(drop=True)
    return df if len(df) >= 50 else None

def simple_features(df):
    df = df.copy()
    df['ret'] = df['close'].pct_change().fillna(0)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14).fillna(50)
    df['sma'] = df['close'].rolling(10).mean().fillna(method='bfill')
    return df.fillna(method='bfill').fillna(0)

def print_progress(current, total, start_time, symbol=""):
    """Print real-time progress with ETA"""
    elapsed = time.time() - start_time
    if current > 0:
        eta = (elapsed / current) * (total - current)
        eta_mins = eta / 60
        progress_pct = (current / total) * 100

        print(f"\r[{progress_pct:5.1f}%] {current:3d}/{total} | "
              f"Elapsed: {elapsed/60:4.1f}m | ETA: {eta_mins:4.1f}m | "
              f"Current: {symbol:<12}", end="", flush=True)
    else:
        print(f"\r[  0.0%]   0/{total} | Starting...", end="", flush=True)

def train_global_model():
    """Train one model on combined data"""
    print("\n🤖 Training global ML model...")
    print("-" * 40)

    X_all, y_all = [], []
    training_symbols = list_symbols()[:20]  # Use 20 symbols for training

    for i, symbol in enumerate(training_symbols):
        print(f"  Training data from {symbol}... ({i+1}/{len(training_symbols)})")

        df = load_data(symbol, '1hour')
        if df is None:
            continue
        df = simple_features(df)

        # Create targets
        y = (df['close'].shift(-1) > df['close']).astype(int)[:-1]
        X = df[['ret', 'rsi', 'sma']].iloc[:-1].values

        mask = ~np.isnan(X).any(axis=1)
        if mask.sum() > 10:  # Need at least 10 valid samples
            X_all.append(X[mask])
            y_all.append(y[mask])

    if not X_all:
        print("❌ No training data available")
        return None, None

    X_combined = np.vstack(X_all)
    y_combined = np.concatenate(y_all)

    print(f"  Training on {len(X_combined):,} samples...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y_combined)

    print(f"✅ Model trained successfully!")
    print(f"   Training samples: {len(X_combined):,}")
    print(f"   Feature count: {X_combined.shape[1]}")
    print()

    return model, scaler

def backtest_symbol(symbol, model, scaler):
    """Backtest one symbol"""
    try:
        df = load_data(symbol, '1hour')
        if df is None:
            return []

        df = simple_features(df)
        trades = []

        for i in range(10, len(df)-1):
            features = df.iloc[i][['ret', 'rsi', 'sma']].values
            if np.isnan(features).any():
                continue

            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            confidence = model.predict_proba(features_scaled)[0][1]

            if prediction == 1 and confidence > 0.65:
                trades.append({
                    'symbol': symbol,
                    'entry_price': df.iloc[i]['close'],
                    'confidence': confidence,
                    'entry_idx': i,
                    'timestamp': datetime.now()
                })

        return trades
    except Exception as e:
        return []  # Skip problematic symbols

if __name__ == "__main__":
    print("🚀 ENSEMBLE BACKTEST WITH PROGRESS TRACKING")
    print("=" * 50)

    start_time = time.time()

    # Step 1: Train model
    model, scaler = train_global_model()
    if model is None:
        print("❌ Failed to train model. Exiting.")
        exit()

    # Step 2: Get all symbols
    symbols = list_symbols()
    total_symbols = len(symbols)

    print(f"📊 Starting backtest on {total_symbols} symbols...")
    print(f"⏰ Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)

    all_trades = []

    # Step 3: Process each symbol with progress tracking
    for i, symbol in enumerate(symbols):
        print_progress(i, total_symbols, start_time, symbol)

        trades = backtest_symbol(symbol, model, scaler)
        all_trades.extend(trades)

        # Save intermediate results every 50 symbols
        if (i + 1) % 50 == 0:
            temp_df = pd.DataFrame(all_trades)
            temp_df.to_csv(f"temp_trades_{i+1}.csv", index=False)

    # Final progress
    print_progress(total_symbols, total_symbols, start_time)
    print()  # New line

    # Step 4: Save final results
    print("\n💾 Saving results...")
    df_trades = pd.DataFrame(all_trades)
    df_trades.to_csv("ensemble_trades_with_progress.csv", index=False)

    # Step 5: Summary
    total_time = time.time() - start_time
    print("\n✅ BACKTEST COMPLETED!")
    print("-" * 25)
    print(f"📊 Total symbols processed: {total_symbols}")
    print(f"🎯 Total trades generated: {len(all_trades):,}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🔄 Average time per symbol: {total_time/total_symbols:.2f} seconds")
    print(f"📁 Results saved to: ensemble_trades_with_progress.csv")

    if len(all_trades) > 0:
        avg_confidence = df_trades['confidence'].mean()
        print(f"📈 Average ML confidence: {avg_confidence:.3f}")

    print(f"\n🏁 Finished at: {datetime.now().strftime('%H:%M:%S')}")
