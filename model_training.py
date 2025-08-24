import numpy as np
import joblib

ML_MODEL_PATH = "model.pkl"
ML_FEATURES = ["rsi", "atr", "adx", "ema10", "ema21", "volumechange"]
ML_THRESHOLD = 0.6

# Load the trained ML model once (outside backtest)
ml_model = joblib.load(ML_MODEL_PATH)

def apply_ml_filter(df, model, features=ML_FEATURES, threshold=ML_THRESHOLD):
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

    # Apply ML filtering to signal candidates before trading loop
    df = apply_ml_filter(df, ml_model)

    for i in range(1, len(df)):
        row = df.iloc[i]
        date, price = row['date'], row['close']
        sig = row.get('entry_signal_final', 0)
        sigtype = 'Long' if sig == 1 else ('Short' if sig == -1 else '')

        # Only proceed if ML filtered the entry signal as True
        ml_entry_ok = row.get('ml_signal', False)
        if sig == 0 or not ml_entry_ok:
            sig = 0  # Do not consider entries that ML model filtered out

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

            # Update highs/lows for trailing stops
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

            # Partial scaling out at first profit target
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

            # Second partial/full scaling at second profit target
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

            # Exit on adaptive ATR stop loss
            if (direction == 1 and price <= adaptive_stop_loss) or (direction == -1 and price >= adaptive_stop_loss):
                reason = 'ATR Stop Loss'

            # Exit on trailing stop
            elif pos.get('trail_active', False) and (
                (direction == 1 and price <= pos.get('trail_stop', 0)) or (direction == -1 and price >= pos.get('trail_stop', 0))
            ):
                reason = 'Trailing Stop'

            # Exit on chandelier exit
            elif (direction == 1 and price <= pos.get('chandelier_exit', 0)) or (direction == -1 and price >= pos.get('chandelier_exit', 0)):
                reason = 'Chandelier Exit'

            # Exit on regime failure
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

        # Entry logic, only if ML-filtered signal is positive
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
