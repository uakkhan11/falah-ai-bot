import os
import pandas as pd
from indicators import add_all_indicators
from strategies_intraday import vwap_bounce_strategy, rsi_breakout_strategy

DATA_DIR = "/root/falah-ai-bot/intraday_data/"
STRATEGIES = {
    "VWAP Bounce": vwap_bounce_strategy,
    "RSI Breakout": rsi_breakout_strategy
}

def run_backtest(file_path, strategy_func):
    df = pd.read_csv(file_path)
    df = add_all_indicators(df)
    df = strategy_func(df)

    trades = []
    for i in range(1, len(df)):
        if df['Signal'].iloc[i]:
            entry = df['close'].iloc[i]
            atr = df['ATR'].iloc[i]
            sl = entry - atr * 1.5
            tp = entry + (entry - sl) * 2
            outcome = None

            for j in range(i+1, min(i+10, len(df))):
                high = df['high'].iloc[j]
                low = df['low'].iloc[j]
                if high >= tp:
                    outcome = "Target Hit"
                    break
                elif low <= sl:
                    outcome = "SL Hit"
                    break
            trades.append((df['date'].iloc[i], entry, sl, tp, outcome))
    return trades

if __name__ == "__main__":
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv"): continue
        file_path = os.path.join(DATA_DIR, file)
        print(f"\n📊 {file}")
        for strat_name, strat_func in STRATEGIES.items():
            trades = run_backtest(file_path, strat_func)
            print(f"📈 {strat_name}: {len(trades)} trades")
            outcome_count = pd.Series([t[-1] for t in trades]).value_counts()
            print(outcome_count.to_string())
