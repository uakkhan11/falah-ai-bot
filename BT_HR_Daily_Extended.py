import os
import pandas as pd
import numpy as np
import talib
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}

YEAR_FILTER = 2025
ML_PROBA_THRESHOLD = 0.5
MIN_TRADE_SIZE = 1
TRADE_SIZE_SCALING = True
VOLUME_MULT_BREAKOUT = 2
ADX_THRESHOLD_BREAKOUT = 25
ADX_THRESHOLD_DEFAULT = 20
ATR_SL_MULT = 1.0
PROFIT_TARGET = 0.02
TRAIL_TRIGGER = 0.005
MAX_TRADES = 500
MAX_POSITIONS = 5
POSITION_SIZE = 10000  # Fixed size per trade
USE_ML_CONFIRM = True
ATR_PERIOD = 14

FEATURES = ["adx", "atr", "volume_ratio", "adosc", "hour_adx", "volume_sma",
            "macd_hist", "vwap", "roc", "obv"]

def load_and_filter_2025(symbol):
    def filter_year(df):
        df['date'] = pd.to_datetime(df['date'])
        return df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)
    daily = pd.read_csv(os.path.join(DATA_PATHS['daily'], f"{symbol}.csv"))
    hourly = pd.read_csv(os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv"))
    m15 = pd.read_csv(os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"))
    daily, hourly, m15 = filter_year(daily), filter_year(hourly), filter_year(m15)
    return daily, hourly, m15

def add_indicators(df):
    if 'date' in df.columns: df = df.sort_values('date')

    df_weekly = df.set_index('date').resample('W-MON').agg({
        'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
    }).dropna().reset_index()

    df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20, 1).max()
    df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high'].reindex(df['date'], method='ffill').values

    df['donchian_high'] = df['high'].rolling(20, 1).max()
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)

    try:
        adx_df = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_df.adx()
    except:
        df['adx'] = np.nan

    df['vol_sma20'] = df['volume'].rolling(20, 1).mean()
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()

    high14 = df['high'].rolling(14).max()
    low14 = df['low'].rolling(14).min()
    df['wpr'] = (high14 - df['close']) / (high14 - low14) * -100

    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=ATR_PERIOD).average_true_range()
    atr_ce = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=22).average_true_range()
    high20 = df['high'].rolling(22, 1).max()
    df['chandelier_exit'] = high20 - 3.0 * atr_ce

    try:
        st = ta.trend.STCIndicator(df['close'], window=10, smooth1=3, smooth2=3)
        df['supertrend'] = st.stc()
        df['supertrend_dir'] = (df['supertrend'] > 50).astype(int)  # simple direction
    except:
        df['supertrend'] = np.nan
        df['supertrend_dir'] = 0

    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)

    df['macd_hist'] = talib.MACD(close)[2]
    df['roc'] = talib.ROC(close, timeperiod=10)
    df['obv'] = talib.OBV(close, volume)
    df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

    df['volume_ratio'] = df['volume'] / df['vol_sma20']
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    return df.reset_index(drop=True)

def add_hourly_features_to_m15(m15_df, hourly_df):
    hourly_df = hourly_df.set_index('date')
    m15_df = m15_df.set_index('date')
    if 'adx' in hourly_df.columns:
        m15_df['hour_adx'] = hourly_df['adx'].reindex(m15_df.index, method='ffill')
    else:
        m15_df['hour_adx'] = np.nan
    return m15_df.reset_index()

def ml_trade_filter(m15_df, hourly_df):
    hourly_df = hourly_df.set_index('date')
    m15_df = m15_df.set_index('date')
    for col in set(['adx','atr','macd_hist']) & set(hourly_df.columns):
        m15_df['hour_' + col] = hourly_df[col].reindex(m15_df.index, method='ffill')
    m15_df['future_return'] = m15_df['close'].shift(-10) / m15_df['close'] - 1
    m15_df['label'] = (m15_df['future_return'] > 0.01).astype(int)
    m15_df.dropna(subset=FEATURES + ['label'], inplace=True)
    X = m15_df[FEATURES]
    y = m15_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'clf_report': classification_report(y_test, y_pred, output_dict=True)
    }
    proba_all = model.predict_proba(X)[:, 1]
    return model, metrics, m15_df.index, proba_all

def apply_ml_filter(df, model):
    if not USE_ML_CONFIRM or model is None:
        df['ml_signal'] = 1
        return df
    features = FEATURES
    df = df.dropna(subset=features).reset_index(drop=True)
    if df.empty:
        return df
    df['ml_signal'] = model.predict(df[features])
    df['entry_signal'] = np.where((df['entry_signal'] == 1) & (df['ml_signal'] == 1), 1, 0)
    return df

def breakout_signal(df):
    cond_d = df['close'] > df['donchian_high'].shift(1)
    cond_v = df['volume'] > VOLUME_MULT_BREAKOUT * df['vol_sma20']
    cond_w = df['close'] > df['weekly_donchian_high'].shift(1)
    df['breakout_signal'] = (cond_d & cond_v & cond_w).astype(int)
    return df

def bb_breakout_signal(df):
    df['bb_breakout_signal'] = ((df['close'] > df['bb_upper']) & (df['volume'] > VOLUME_MULT_BREAKOUT * df['vol_sma20'])).astype(int)
    return df

def bb_pullback_signal(df):
    cond_pull = df['close'] < df['bb_lower']
    cond_resume = df['close'] > df['bb_lower'].shift(1)
    df['bb_pullback_signal'] = (cond_pull.shift(1) & cond_resume).astype(int)
    return df

def combine_signals(df):
    chand_or_st = (df['close'] > df['chandelier_exit']) | (df['supertrend_dir'] == 1)
    regime_breakout = (df['close'] > df['ema200']) & (df['adx'] > ADX_THRESHOLD_BREAKOUT)
    regime_default = (df['close'] > df['ema200']) & (df['adx'] > ADX_THRESHOLD_DEFAULT)
    df['entry_signal'] = 0
    df['entry_type'] = ''
    df.loc[(df['breakout_signal'] == 1) & chand_or_st & regime_breakout,
           ['entry_signal', 'entry_type']] = [1, 'Breakout']
    df.loc[(df['bb_breakout_signal'] == 1) & chand_or_st & regime_breakout & (df['entry_signal'] == 0),
           ['entry_signal', 'entry_type']] = [1, 'BB_Breakout']
    df.loc[(df['bb_pullback_signal'] == 1) & chand_or_st & regime_default & (df['entry_signal'] == 0),
           ['entry_signal', 'entry_type']] = [1, 'BB_Pullback']
    return df

def backtest(df, symbol):
    INITIAL_CAPITAL = 100000
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    trade_count = 0
    regime_fail_count = {}
    for i in range(1, len(df)):
        row = df.iloc[i]
        date, price, sig, sigtype = row['date'], row['close'], row['entry_signal'], row['entry_type']
        regime_ok = (row['close'] > row['ema200']) and (row['adx'] > ADX_THRESHOLD_DEFAULT)
        # EXIT logic
        to_close = []
        for pid, pos in positions.items():
            ret = (price - pos['entry_price']) / pos['entry_price']
            atr_stop = pos['entry_price'] - ATR_SL_MULT * pos['entry_atr']
            if price > pos['high']:
                pos['high'] = price
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = row['chandelier_exit']
            if pos['trail_active'] and row['chandelier_exit'] > pos['trail_stop']:
                pos['trail_stop'] = row['chandelier_exit']
            if ret >= PROFIT_TARGET:
                reason = 'Profit Target'
            elif price <= atr_stop:
                reason = 'ATR Stop Loss'
            elif pos['trail_active'] and price <= pos['trail_stop']:
                reason = 'Chandelier Exit'
            else:
                pid_key = f"{symbol}_{pid}"
                if not regime_ok:
                    regime_fail_count[pid_key] = regime_fail_count.get(pid_key, 0) + 1
                else:
                    regime_fail_count[pid_key] = 0
                if regime_fail_count.get(pid_key, 0) >= 2:
                    reason = 'Regime Exit'
                else:
                    reason = None
            if reason:
                buy_val = pos['shares'] * pos['entry_price']
                sell_val = pos['shares'] * price
                charges = 0.0005 * (buy_val + sell_val)
                pnl = sell_val - buy_val - charges
                trades.append({'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': date,
                               'pnl': pnl, 'entry_type': pos['entry_type'], 'exit_reason': reason})
                cash += sell_val
                to_close.append(pid)
                trade_count += 1
                if trade_count >= MAX_TRADES:
                    break
        for pid in to_close:
            positions.pop(pid)
        if trade_count >= MAX_TRADES:
            break
        # ENTRY logic
        if sig == 1 and len(positions) < MAX_POSITIONS and cash >= POSITION_SIZE:
            shares = POSITION_SIZE / price
            positions[len(positions) + 1] = {'entry_date': date, 'entry_price': price,
                                           'shares': shares, 'high': price, 'trail_active': False,
                                           'trail_stop': 0, 'entry_atr': row['atr'], 'entry_type': sigtype}
            cash -= POSITION_SIZE
    return trades

def extract_trade_stats(trades):
    df = pd.DataFrame(trades)
    if df.empty or 'pnl' not in df.columns:
        return {}
    df['win'] = df['pnl'] > 0
    stats = {
        'Total Trades': len(df),
        'Winning Trades': df['win'].sum(),
        'Losing Trades': len(df) - df['win'].sum(),
        'Win Rate %': round(df['win'].mean() * 100, 2),
        'Avg PnL per Trade': round(df['pnl'].mean(), 2),
        'Best PnL': round(df['pnl'].max(), 2),
        'Worst PnL': round(df['pnl'].min(), 2),
        'Total PnL': round(df['pnl'].sum(), 2)
    }
    return stats

def summarize_strategy_performance(df):
    strategies = df['Strategy'].unique()
    summary = []
    for strat in strategies:
        strat_df = df[df['Strategy'] == strat]
        traded_symbols = strat_df['Symbol'].nunique()
        profitable_symbols = strat_df[strat_df['Total PnL'] > 0]['Symbol'].nunique()
        losing_symbols = strat_df[strat_df['Total PnL'] <= 0]['Symbol'].nunique()
        avg_win_rate = strat_df['Win Rate %'].mean()
        total_pnl = strat_df['Total PnL'].sum()
        summary.append({
            'Strategy': strat,
            'Symbols Traded': traded_symbols,
            'Profitable Symbols': profitable_symbols,
            'Losing Symbols': losing_symbols,
            'Average Win Rate %': round(avg_win_rate, 2),
            'Total PnL': round(total_pnl, 2)
        })
    summary_df = pd.DataFrame(summary)
    print("\n--- Strategy Performance Summary ---")
    print(summary_df)
    return summary_df

def get_symbols_from_data():
    daily_files = os.listdir(DATA_PATHS['daily'])
    return [os.path.splitext(f)[0] for f in daily_files if f.endswith('.csv')]

if __name__ == "__main__":
    symbols = get_symbols_from_data()
    all_stats = []
    for symbol in symbols:
        daily, hourly, m15 = prepare_data_2025(symbol)
        ml_model, ml_metrics, ml_index, ml_proba = ml_trade_filter(m15, hourly)
        m15 = apply_ml_filter(m15, ml_model)
        trades_ml = backtest(m15, symbol)
        stats_ml = extract_trade_stats(trades_ml)
        stats_ml.update({
            'Symbol': symbol,
            'Strategy': 'ML+All Features',
            'ML Accuracy': round(ml_metrics['accuracy'], 4),
            'ML Precision': round(ml_metrics['precision'], 4),
            'ML Recall': round(ml_metrics['recall'], 4)
        })
        all_stats.append(stats_ml)
    df_summary = pd.DataFrame(all_stats)
    df_summary.to_csv("final_ml_only_backtest_report.csv", index=False)
    summarize_strategy_performance(df_summary)
