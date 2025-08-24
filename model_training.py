import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report

# --- Config ---
CSV_PATH = "your_training_data.csv"  # Update your training data CSV path here
MODEL_SAVE_PATH = "model.pkl"
TARGET_COLUMN = "outcome"
FEATURES = ["rsi", "atr", "adx", "ema10", "ema21", "volumechange"]
ML_THRESHOLD = 0.6

DATA_DIR_DAILY = "/root/falah-ai-bot/swing_data"
DATA_DIR_1H = "/root/falah-ai-bot/intraday_swing_data"
DATA_DIR_15M = "/root/falah-ai-bot/scalping_data"
SYMBOLS = ["RELIANCE", "SUNPHARMA"]
YEARS_BACK = 2

# --- ML Model Training & Loading ---
def train_and_save_model():
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Model '{MODEL_SAVE_PATH}' found. Loading existing model.")
        return joblib.load(MODEL_SAVE_PATH)

    print("Training new model from scratch...")
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.lower() for c in df.columns]
    df.dropna(subset=FEATURES + [TARGET_COLUMN], inplace=True)

    X = df[FEATURES]
    y = df[TARGET_COLUMN].astype(int)

    X_sample, _, y_sample, _ = train_test_split(
        X, y, test_size=0.9, stratify=y, random_state=42)

    param_dist = {
        "n_estimators": [100, 150],
        "max_depth": [None, 10],
        "min_samples_split": [2],
        "class_weight": ["balanced"],
    }

    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    random_search.fit(X_sample, y_sample)
    print(f"Best params: {random_search.best_params_}")

    final_model = RandomForestClassifier(**random_search.best_params_, random_state=42)
    final_model.fit(X, y)

    cv_scores = cross_val_score(final_model, X, y, cv=5, scoring="f1")
    print(f"5-fold CV F1 score: {cv_scores.mean():.4f}")

    y_pred = final_model.predict(X)
    print("Classification report on training data:")
    print(classification_report(y, y_pred))

    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"Saved trained model to {MODEL_SAVE_PATH}")

    return final_model

# --- Data loading & indicator functions ---
def add_weekly_ema(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df_weekly = df.set_index('date').resample('W-MON')['close'].last().dropna().to_frame()
    df_weekly['ema50'] = ta.ema(df_weekly['close'], length=50)
    df_weekly['ema50_slope'] = df_weekly['ema50'].diff()
    df = df.set_index('date')
    df['weekly_ema50'] = df_weekly['ema50'].reindex(df.index, method='ffill')
    df['weekly_ema50_slope'] = df_weekly['ema50_slope'].reindex(df.index, method='ffill')
    df = df.reset_index()
    return df

def load_candle_data(symbol, timeframe, years=YEARS_BACK):
    folder_map = {
        "daily": DATA_DIR_DAILY,
        "1h": DATA_DIR_1H,
        "15m": DATA_DIR_15M,
    }
    path = os.path.join(folder_map[timeframe], f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"Data file missing for {symbol} timeframe {timeframe}: {path}")
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365 * years)
    df = df[df['date'] >= cutoff].sort_values('date').reset_index(drop=True)
    return df

def prepare_multitimeframe_data(symbol):
    df_daily = load_candle_data(symbol, "daily")
    df_1h = load_candle_data(symbol, "1h")
    df_15m = load_candle_data(symbol, "15m")
    if df_daily is None or df_1h is None or df_15m is None:
        return None

    df_15m['rsi_15m'] = ta.rsi(df_15m['close'], length=14)
    df_15m['ema20_15m'] = ta.ema(df_15m['close'], length=20)
    df_15m['ema20_15m_slope'] = df_15m['ema20_15m'].diff()

    df_1h['rsi_1h'] = ta.rsi(df_1h['close'], length=14)
    df_1h['ema50_1h'] = ta.ema(df_1h['close'], length=50)
    df_1h['ema50_1h_slope'] = df_1h['ema50_1h'].diff()

    daily_15m_agg = df_15m.groupby(df_15m['date'].dt.floor('D')).agg({
        'rsi_15m': 'last',
        'ema20_15m_slope': 'last'
    }).rename_axis('date').reset_index()

    daily_1h_agg = df_1h.groupby(df_1h['date'].dt.floor('D')).agg({
        'rsi_1h': 'last',
        'ema50_1h_slope': 'last'
    }).rename_axis('date').reset_index()

    df_daily = pd.merge(df_daily, daily_15m_agg, on='date', how='left')
    df_daily = pd.merge(df_daily, daily_1h_agg, on='date', how='left')

    df_daily['rsi_15m'] = df_daily['rsi_15m'].ffill()
    df_daily['ema20_15m_slope'] = df_daily['ema20_15m_slope'].fillna(0)
    df_daily['rsi_1h'] = df_daily['rsi_1h'].ffill()
    df_daily['ema50_1h_slope'] = df_daily['ema50_1h_slope'].fillna(0)

    df_daily = add_indicators(df_daily)  # Your existing function needs to be available
    df_daily = add_weekly_ema(df_daily)

    return df_daily

def modify_combine_signals_with_mtf(df):
    df = combine_signals(df)  # Your existing function needs to be available or implement similarly
    long_cond = (
        (df['entry_signal'] == 1) &
        (df['rsi_15m'] > 50) &
        (df['rsi_1h'] > 50) &
        (df['ema20_15m_slope'] > 0) &
        (df['ema50_1h_slope'] > 0)
    )
    short_cond = (
        (df['close'] < df['ema200']) &
        (df['rsi_15m'] < 50) &
        (df['rsi_1h'] < 50) &
        (df['ema20_15m_slope'] < 0) &
        (df['ema50_1h_slope'] < 0)
    )
    df['entry_signal_long'] = long_cond
    df['entry_signal_short'] = short_cond
    df['entry_signal_final'] = 0
    df.loc[long_cond, 'entry_signal_final'] = 1
    df.loc[short_cond, 'entry_signal_final'] = -1
    return df

def apply_ml_filter(df, model, features=FEATURES, threshold=ML_THRESHOLD):
    X = df[features]
    missing_mask = X.isnull().any(axis=1)
    preds_prob = np.zeros(len(df))
    if not missing_mask.all():
        preds_prob_valid = model.predict_proba(X[~missing_mask])[:, 1]
        preds_prob[~missing_mask] = preds_prob_valid
    df['ml_prob'] = preds_prob
    df['ml_signal'] = (df['ml_prob'] >= threshold) & (df['entry_signal_final'] != 0)
    return df

def backtest_mtf(df, symbol):
    INITIAL_CAPITAL = 1_000_000
    RISK_PER_TRADE = 0.01 * INITIAL_CAPITAL
    PROFIT_TARGET1 = 0.10
    PROFIT_TARGET2 = 0.15
    ATR_SL_MULT = 2.8
    TRAIL_TRIGGER = 0.07
    MAX_POSITIONS = 5
    MAX_TRADES = 2000
    TRANSACTION_COST = 0.001
    RSI_THRESHOLD = 55
    EMA_SLOPE_THRESHOLD = 0.0

    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    trade_count = 0
    regime_fail_count = {}

    rolling_atr_mean = df['atr'].rolling(window=20, min_periods=1).mean()

    df = apply_ml_filter(df, joblib.load(MODEL_SAVE_PATH))

    for i in range(1, len(df)):
        row = df.iloc[i]
        date, price = row['date'], row['close']
        sig = row.get('entry_signal_final', 0)
        sigtype = 'Long' if sig == 1 else ('Short' if sig == -1 else '')

        ml_entry_ok = row.get('ml_signal', False)
        if sig == 0 or not ml_entry_ok:
            sig = 0

        regime_ok = (
            (df.at[i, 'ema200_slope'] > EMA_SLOPE_THRESHOLD) and
            (row['rsi'] > RSI_THRESHOLD)
        )

        if i > 0 and not pd.isna(row['atr']):
            stop_loss_distance = ATR_SL_MULT * row['atr']
        else:
            stop_loss_distance = ATR_SL_MULT * (df['atr'].mean() if not df['atr'].isna().all() else 1)

        position_size = min(cash, RISK_PER_TRADE / stop_loss_distance * price)

        to_close = []
        for pid, pos in list(positions.items()):
            direction = pos.get('direction', 1)
            ret = direction * (price - pos['entry_price']) / pos['entry_price']

            adaptive_atr_mult = ATR_SL_MULT * (rolling_atr_mean.iloc[i] / rolling_atr_mean.mean())
            adaptive_stop_loss = pos['entry_price'] - direction * adaptive_atr_mult * pos.get('entry_atr', 0)

            if (direction == 1 and price > pos['high']) or (direction == -1 and price < pos['low']):
                if direction == 1:
                    pos['high'] = price
                else:
                    pos['low'] = price

            if not pos.get('trail_active', False) and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = row.get('chandelier_exit', 0)

            if pos.get('trail_active', False):
                if (direction == 1 and row.get('chandelier_exit', 0) > pos.get('trail_stop', 0)) or \
                   (direction == -1 and row.get('chandelier_exit', 0) < pos.get('trail_stop', 0)):
                    pos['trail_stop'] = row.get('chandelier_exit', 0)

            reason = None
            pnl = 0

            if ret >= PROFIT_TARGET1 and not pos.get('scale1', False):
                scale_qty = pos['shares'] * 0.5
                remain_qty = pos['shares'] - scale_qty
                buy_val = scale_qty * pos['entry_price']
                sell_val = scale_qty * price
                charges = (buy_val + sell_val) * TRANSACTION_COST
                pnl = direction * (sell_val * (1 - TRANSACTION_COST) - buy_val) - charges

                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'entry_type': pos['entry_type'],
                    'exit_reason': 'Partial Profit Target 1'
                })

                pos['shares'] = remain_qty
                pos['scale1'] = True
                cash += sell_val
                continue

            if ret >= PROFIT_TARGET2 and not pos.get('scale2', False):
                scale_qty = pos['shares']
                buy_val = scale_qty * pos['entry_price']
                sell_val = scale_qty * price
                charges = (buy_val + sell_val) * TRANSACTION_COST
                pnl = direction * (sell_val * (1 - TRANSACTION_COST) - buy_val) - charges

                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'entry_type': pos['entry_type'],
                    'exit_reason': 'Partial Profit Target 2'
                })

                cash += sell_val
                to_close.append(pid)
                trade_count += 1
                continue

            if (direction == 1 and price <= adaptive_stop_loss) or (direction == -1 and price >= adaptive_stop_loss):
                reason = 'ATR Stop Loss'
            elif pos.get('trail_active', False) and (
                (direction == 1 and price <= pos.get('trail_stop', 0)) or (direction == -1 and price >= pos.get('trail_stop', 0))
            ):
                reason = 'Trailing Stop'
            elif (direction == 1 and price <= pos.get('chandelier_exit', 0)) or (direction == -1 and price >= pos.get('chandelier_exit', 0)):
                reason = 'Chandelier Exit'
            else:
                pid_key = f"{symbol}_{pid}"
                if not regime_ok:
                    regime_fail_count[pid_key] = regime_fail_count.get(pid_key, 0) + 1
                else:
                    regime_fail_count[pid_key] = 0
                if regime_fail_count.get(pid_key, 0) >= 2:
                    reason = 'Regime Exit'

            if reason:
                buy_val = pos['shares'] * pos['entry_price']
                sell_val = pos['shares'] * price
                charges = (buy_val + sell_val) * TRANSACTION_COST
                pnl = direction * (sell_val * (1 - TRANSACTION_COST) - buy_val) - charges

                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'entry_type': pos['entry_type'],
                    'exit_reason': reason
                })

                cash += sell_val
                to_close.append(pid)
                trade_count += 1

                if trade_count >= MAX_TRADES:
                    break

        for pid in to_close:
            if pid in positions:
                del positions[pid]

        if trade_count >= MAX_TRADES:
            break

        if sig != 0 and row.get('ml_signal', False) and len(positions) < MAX_POSITIONS and cash >= position_size:
            shares = position_size / price
            direction = 1 if sig == 1 else -1

            positions[len(positions) + 1] = {
                'entry_date': date,
                'entry_price': price,
                'shares': shares,
                'high': price,
                'low': price,
                'trail_active': False,
                'trail_stop': 0,
                'entry_atr': row.get('atr', 0),
                'entry_type': sigtype,
                'scale1': False,
                'scale2': False,
                'direction': direction,
            }
            cash -= position_size * (1 + TRANSACTION_COST)

    return trades

def main():
    # Train or load ML model
    ml_model = train_and_save_model()

    for symbol in SYMBOLS:
        print(f"\nBacktesting {symbol} multi-timeframe with ML filtering for last {YEARS_BACK} years...")
        df = prepare_multitimeframe_data(symbol)
        if df is None:
            print(f"Skipping {symbol} due to missing data.")
            continue
        df = modify_combine_signals_with_mtf(df)
        df = apply_ml_filter(df, ml_model)
        trades = backtest_mtf(df, symbol)

        if trades:
            trades_df = pd.DataFrame(trades)
            print(f"{symbol} Backtest Results:")
            print(f"Total trades: {len(trades_df)}")
            print(f"Total PnL: {trades_df['pnl'].sum():.2f}")
            print(f"Win rate: {(trades_df['pnl'] > 0).mean() * 100:.2f}%")
            print("\nPnL by Entry Type:")
            print(trades_df.groupby('entry_type')['pnl'].agg(['count', 'sum', 'mean']))
            print("\nExit Reason Performance:")
            print(trades_df.groupby('exit_reason')['pnl'].agg(['count', 'sum', 'mean']))
        else:
            print(f"No trades executed for {symbol}.")

if __name__ == "__main__":
    main()
