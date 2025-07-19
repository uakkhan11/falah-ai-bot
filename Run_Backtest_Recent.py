import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from indicators import add_indicators
from ai_engine import get_ai_score

DATA_FOLDER = '/root/falah-ai-bot/historical_data/'
start_date = datetime.now(timezone.utc) - timedelta(days=180)  # 6 months
filter_start_date = datetime.now(timezone.utc) - timedelta(days=90)  # 3 months for trade filter

symbol_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
print(f"✅ Total symbols found: {len(symbol_files)}")

valid_symbols = []
total_trades = 0

for file in symbol_files:
    symbol = file.replace('.csv', '')
    file_path = os.path.join(DATA_FOLDER, file)

    try:
        df = pd.read_csv(file_path)

        # Validate columns
        if not {'date', 'open', 'high', 'low', 'close', 'volume'}.issubset(df.columns):
            print(f"⚠️ Skipping {symbol}: Missing required columns")
            continue

        # Prepare dataframe
        df.rename(columns={"date": "datetime"}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.sort_values('datetime')
        df = df[df['datetime'] >= start_date].reset_index(drop=True)

        if len(df) < 30:
            print(f"⚠️ Skipping {symbol}: Not enough data after filtering")
            continue

        df = add_indicators(df)

        trades = 0
        for idx, row in df.iterrows():
            dt = row['datetime']
            if dt < filter_start_date:
                continue

            subset_df = df.iloc[:idx + 1].copy()
            ai_score = get_ai_score(subset_df)

            if ai_score is None:
                continue

            if ai_score > 0.25:
                print(f"✅ {symbol} | BUY | {dt.date()} | Close={row['close']:.2f} | AI Score={ai_score:.2f}")
                trades += 1

        if trades > 0:
            valid_symbols.append(symbol)
            total_trades += trades

    except Exception as e:
        print(f"⚠️ Skipping {symbol}: Error -> {e}")

print("\n===== BACKTEST SUMMARY =====")
print(f"✅ Valid symbols with trades: {len(valid_symbols)}")
print(f"✅ Total trades executed: {total_trades}")
print("===== BACKTEST COMPLETE =====")
