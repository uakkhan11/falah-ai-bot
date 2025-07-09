import os
import pandas as pd
import numpy as np
from datetime import datetime
from ai_engine import calculate_ai_exit_score

# ───── CONFIG ─────────────────────────────────────────────────
DATA_DIR       = "./historical_data"
SYMBOLS        = ["INFY", "TCS", "HDFCBANK"]
START_DATE     = "2018-01-01"
END_DATE       = "2023-12-31"
INITIAL_EQUITY = 1_000_000
RISK_PER_TRADE = 0.02    # 2% of current equity
COMMISSION     = 0.0005  # 0.05%
SLIPPAGE       = 0.0005  # 0.05%

# ───── STORAGE ────────────────────────────────────────────────
equity       = INITIAL_EQUITY
peak_equity  = INITIAL_EQUITY
drawdowns    = []
equity_curve = []
all_trades   = []

# ───── STRATEGY INTERFACE ─────────────────────────────────────
class BaseStrategy:
    def generate_signals(self, df):
        raise NotImplementedError

# ───── EXAMPLE STRATEGIES ─────────────────────────────────────
class EMACrossoverStrategy(BaseStrategy):
    def generate_signals(self, df):
        df = df.copy()
        df['EMA10'] = df['close'].ewm(span=10).mean()
        df['EMA21'] = df['close'].ewm(span=21).mean()
        signals, in_trade = [], False
        for i in range(21, len(df)):
            date, price = df.index[i], df['close'].iat[i]
            if not in_trade and df['EMA10'].iat[i] > df['EMA21'].iat[i]:
                sl = price * 0.98
                tp = price * 1.06
                signals.append({'entry_date': date, 'entry': price, 'sl': sl, 'tp': tp})
                in_trade = True
            elif in_trade:
                low, high = df['low'].iat[i], df['high'].iat[i]
                exit_price = None
                if low <= signals[-1]['sl']:
                    exit_price = signals[-1]['sl']
                elif high >= signals[-1]['tp']:
                    exit_price = signals[-1]['tp']
                if exit_price is not None:
                    signals[-1].update({'exit_date': date, 'exit': exit_price})
                    in_trade = False
        return signals

class RSIStrategy(BaseStrategy):
    def generate_signals(self, df):
        df = df.copy()
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        df['RSI'] = 100 - 100/(1 + gain/loss)
        signals, in_trade = [], False
        for i in range(14, len(df)):
            date, price, rsi = df.index[i], df['close'].iat[i], df['RSI'].iat[i]
            if not in_trade and rsi < 30:
                sl = price * 0.98
                tp = price * 1.06
                signals.append({'entry_date': date, 'entry': price, 'sl': sl, 'tp': tp})
                in_trade = True
            elif in_trade and rsi > 50:
                signals[-1].update({'exit_date': date, 'exit': price})
                in_trade = False
        return signals

class VolumeBreakoutStrategy(BaseStrategy):
    def generate_signals(self, df):
        df = df.copy()
        df['vol_avg'] = df['volume'].rolling(20).mean()
        signals, in_trade = [], False
        for i in range(20, len(df)):
            date, price = df.index[i], df['close'].iat[i]
            vol, vol_avg = df['volume'].iat[i], df['vol_avg'].iat[i]
            if not in_trade and vol > 1.5 * vol_avg:
                sl = price * 0.98
                tp = price * 1.06
                signals.append({'entry_date': date, 'entry': price, 'sl': sl, 'tp': tp})
                in_trade = True
            elif in_trade:
                low, high = df['low'].iat[i], df['high'].iat[i]
                exit_price = None
                if low <= signals[-1]['sl']:
                    exit_price = signals[-1]['sl']
                elif high >= signals[-1]['tp']:
                    exit_price = signals[-1]['tp']
                if exit_price:
                    signals[-1].update({'exit_date': date, 'exit': exit_price})
                    in_trade = False
        return signals

class AIScoreExitStrategy(BaseStrategy):
    def __init__(self, ai_exit_threshold=70):
        self.threshold = ai_exit_threshold
    def generate_signals(self, df):
        df = df.copy()
        df['SMA20'] = df['close'].rolling(20).mean()
        signals, in_trade = [], False
        for i in range(20, len(df)):
            date, price = df.index[i], df['close'].iat[i]
            if not in_trade and price > df['SMA20'].iat[i]:
                atr = df['high'].sub(df['low']).rolling(14).mean().iat[i]
                trailing_sl = price - 1.5 * atr
                signals.append({'entry_date': date, 'entry': price, 'sl': trailing_sl, 'tp': None})
                in_trade = True
            elif in_trade:
                hist = df.iloc[:i+1].reset_index()
                last = signals[-1]
                ai_score, _ = calculate_ai_exit_score(
                    stock_data=hist,
                    trailing_sl=last['sl'],
                    current_price=price,
                    atr_value=atr
                )
                if ai_score >= self.threshold:
                    signals[-1].update({'exit_date': date, 'exit': price})
                    in_trade = False
        return signals

# ───── BACKTEST ENGINE ────────────────────────────────────────
def backtest():
    global equity, peak_equity
    strategies = [
        EMACrossoverStrategy(),
        RSIStrategy(),
        VolumeBreakoutStrategy(),
        AIScoreExitStrategy(ai_exit_threshold=70)
    ]

    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{sym}.csv")
        if not os.path.exists(path):
            print(f"⚠️ Missing data for {sym}")
            continue
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = df.loc[START_DATE:END_DATE]

        for strat in strategies:
            trades = strat.generate_signals(df)
            for t in trades:
                entry, sl, tp = t['entry'], t['sl'], t['tp']
                risk_amount = equity * RISK_PER_TRADE
                qty = int(risk_amount / max(entry - sl, 1e-6))
                if qty < 1:
                    continue
                buy_price = entry * (1 + SLIPPAGE)
                cost = buy_price * qty * (1 + COMMISSION)
                exit_price = t.get('exit')
                if exit_price is None:
                    next_idx = df.index.get_loc(t['entry_date']) + 1
                    if next_idx < len(df):
                        exit_price = df['close'].iat[next_idx]
                    else:
                        continue
                sell_price = exit_price * (1 - SLIPPAGE)
                proceeds = sell_price * qty * (1 - COMMISSION)
                pnl = proceeds - cost
                equity += pnl
                peak_equity = max(peak_equity, equity)
                drawdowns.append((peak_equity - equity) / peak_equity)
                exit_date = t['exit_date'] if t.get('exit_date') else df.index[next_idx]
                equity_curve.append({'date': exit_date, 'equity': equity})
                all_trades.append({
                    'symbol': sym,
                    'strategy': strat.__class__.__name__,
                    'entry_date': t['entry_date'],
                    'exit_date': exit_date,
                    'qty': qty,
                    'entry': buy_price,
                    'exit': sell_price,
                    'pnl': pnl
                })

    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv("all_trades.csv", index=False)
    ec = pd.DataFrame(equity_curve)
    if 'date' not in ec.columns:
        raise KeyError("Equity curve records missing 'date' field")
    ec = ec.set_index('date')
    ec.to_csv("equity_curve.csv")

    returns = ec['equity'].pct_change().dropna()
    days = (ec.index[-1] - ec.index[0]).days / 365.25
    cagr = (equity / INITIAL_EQUITY) ** (1 / days) - 1 if days>0 else np.nan
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else np.nan
    max_dd = max(drawdowns) * 100 if drawdowns else 0.0
    print(f"CAGR: {cagr:.2%}  |  Sharpe: {sharpe:.2f}  |  Max DD: {max_dd:.1f}%")

if __name__ == "__main__":
    backtest()
