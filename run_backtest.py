# run_backtest.py

import os
import pandas as pd
import joblib
import pandas_ta as ta
from datetime import datetime

# ===== CONFIG =====
HISTORICAL_DIR = "/root/falah-ai-bot/historical_data/"
MODEL_FILE = "model.pkl"
OUTPUT_TRADES = "backtest_trades.csv"

# Load AI model
model = joblib.load(MODEL_FILE)

def calculate_features(df):
    """Calculate all features needed for entry signals."""
    df['EMA10'] = ta.ema(df['close'], length=10)
    df['EMA21'] = ta.ema(df['close'], length=21)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['VolumeChange'] = df['volume'].pct_change().fillna(0)
    st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
    df['Supertrend'] = st['SUPERT_10_3.0']
    return df

def apply_ai_score(df):
    features = ["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange"]
    df = df.dropna(subset=features)
    X = df[features]
    df['ai_score'] = model.predict_proba(X)[:, 1]
    return df
    
def run_backtest():
    trades = []
    initial_capital = 1_000_000
    capital = initial_capital
    position = None

    for file in sorted(os.listdir(HISTORICAL_DIR)):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(HISTORICAL_DIR, file))
        if df.shape[0] < 100:
            continue

        df.columns = [c.lower() for c in df.columns]
        df = calculate_features(df)
        df = apply_ai_score(df)

        for i in range(50, len(df)):  # start after enough candles
            row = df.iloc[i]

            # Exit logic if in position
            if position and position['symbol'] == symbol and not pd.isna(row['close']):
                ltp = row['close']
                atr = row['ATR']
                trailing_sl_atr = ltp - 1.5 * atr
                trailing_sl_recent = df['low'].iloc[i-7:i].min()
                trailing_sl = max(trailing_sl_atr, trailing_sl_recent)

                ai_score_exit = 0
                reasons = []

                if ltp < position['entry_price'] * 0.98:
                    ai_score_exit += 25; reasons.append("Fixed SL breach (-2%)")
                if ltp < trailing_sl:
                    ai_score_exit += 20; reasons.append("Trailing SL breached")
                if ltp >= position['entry_price'] * 1.12:
                    ai_score_exit += 10; reasons.append("Profit >=12% hit")

                if ai_score_exit >= 30 or ltp >= position['target_price']:
                    # Full exit
                    exit_value = position['qty'] * ltp
                    pnl = exit_value - (position['qty'] * position['entry_price'])
                    capital += exit_value
                    trades.append({
                        "date": row['date'], "symbol": symbol,
                        "entry_price": position['entry_price'],
                        "exit_price": ltp, "qty": position['qty'],
                        "pnl": pnl, "reason": ",".join(reasons) or "Target hit"
                    })
                    position = None
                    continue

            # Entry logic if not in position
            if not position:
                if (35 < row['RSI'] < 65 and
                    row['EMA10'] > row['EMA21'] and
                    row['ai_score'] > 0.25 and
                    row['close'] > row['Supertrend']):
                    
                    risk_per_trade = 0.02 * capital
                    sl_price = row['close'] - 2 * row['ATR']
                    qty = int(risk_per_trade / (row['close'] - sl_price))

                    if qty > 0:
                        capital -= qty * row['close']
                        position = {
                            "symbol": symbol,
                            "entry_price": row['close'],
                            "qty": qty,
                            "target_price": row['close'] * 1.12
                        }

    # Save trades
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(OUTPUT_TRADES, index=False)

    # Summary
    total_trades = len(trades_df)
    profitable = trades_df[trades_df['pnl'] > 0].shape[0]
    win_rate = (profitable / total_trades * 100) if total_trades > 0 else 0
    total_pnl = trades_df['pnl'].sum()

    print("\n===== BACKTEST SUMMARY =====")
    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {profitable} ({win_rate:.2f}%)")
    print(f"Total PnL: ₹{total_pnl:,.2f}")
    print(f"Final Capital: ₹{capital:,.2f}")
    print("============================\n")

if __name__ == "__main__":
    run_backtest()
