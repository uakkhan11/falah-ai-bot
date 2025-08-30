import os
import pandas as pd
import numpy as np
import talib
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

def load_and_filter_2025(symbol):
    def filter_year(df):
        df['date'] = pd.to_datetime(df['date'])
        return df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)
    daily = pd.read_csv(os.path.join(DATA_PATHS['daily'], f"{symbol}.csv"))
    hourly = pd.read_csv(os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv"))
    m15 = pd.read_csv(os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"))
    daily, hourly, m15 = filter_year(daily), filter_year(hourly), filter_year(m15)
    return daily, hourly, m15

def compute_indicators(df):
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill()
    df['high'] = pd.to_numeric(df['high'], errors='coerce').ffill()
    df['low'] = pd.to_numeric(df['low'], errors='coerce').ffill()
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

    df['ema8'] = talib.EMA(close, timeperiod=8)
    df['ema20'] = talib.EMA(close, timeperiod=20)
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    df['macd_hist'] = talib.MACD(close)[2]
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['roc'] = talib.ROC(close, timeperiod=10)
    df['obv'] = talib.OBV(close, volume)
    df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['volume_sma'] = df['volume'].rolling(14).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['hour_adx'] = df['adx'] if 'adx' in df else np.nan
    df['highest_high_22'] = df['high'].rolling(window=22).max()
    df['chandelier_exit'] = df['highest_high_22'] - 3 * df['atr']

    # Breakout & Pullback flags
    df['breakout'] = df['close'] > df['high'].rolling(window=20).max().shift(1)
    df['pullback'] = (df['close'] < df['ema20']) & (df['ema8'] > df['ema20'])

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def prepare_data_2025(symbol):
    daily, hourly, m15 = load_and_filter_2025(symbol)
    daily, hourly, m15 = compute_indicators(daily), compute_indicators(hourly), compute_indicators(m15)
    daily.dropna(subset=['ema8','ema20'], inplace=True)
    hourly.dropna(subset=['ema8','ema20'], inplace=True)
    m15.dropna(subset=['ema8','ema20'], inplace=True)
    return daily, hourly, m15

FEATURES = ["adx", "atr", "volume_ratio", "adosc", "hour_adx", "volume_sma",
            "macd_hist", "vwap", "roc", "obv"]

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
    proba_all = model.predict_proba(X)[:,1]
    return model, metrics, m15_df.index, proba_all

class Backtest2025Next:
    def __init__(self, daily_df, hourly_df, m15_df, ml_model=None, ml_proba=None, ml_index=None, init_cap=100000,
                 mode='ml_all'):
        """
        mode options:
        - 'ml_all'   : use ML + indicator logic (original)
        - 'breakout' : only enter trades flagged breakout==True
        - 'pullback' : only enter trades flagged pullback==True
        """
        self.daily_df = daily_df
        self.hourly_df = hourly_df
        self.m15_df = m15_df
        self.ml_model = ml_model
        self.ml_proba = pd.Series(ml_proba, index=ml_index) if ml_proba is not None else None
        self.cash = init_cap
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.highest_price = 0
        self.mode = mode

    def run(self):
        if self.m15_df.empty:
            return []
        for i in range(1, len(self.m15_df)):
            curr = self.m15_df.iloc[i]
            curr_idx = curr.name

            if self.mode == 'ml_all':
                trade_ml_prob = self.ml_proba.get(curr_idx, 0)
                daily_ok = (self.daily_df['ema8'].iloc[-1] > self.daily_df['ema20'].iloc[-1])
                hourly_ok = (self.hourly_df['ema8'].iloc[-1] > self.hourly_df['ema20'].iloc[-1])
                can_enter = daily_ok and hourly_ok and curr['adx'] > 20 and trade_ml_prob > ML_PROBA_THRESHOLD
            elif self.mode == 'breakout':
                can_enter = curr['breakout'] == True
            elif self.mode == 'pullback':
                can_enter = curr['pullback'] == True
            else:
                can_enter = False

            if self.position == 0:
                if can_enter:
                    max_qty = int(self.cash / curr['close'])
                    qty = MIN_TRADE_SIZE
                    if TRADE_SIZE_SCALING and self.mode == 'ml_all':
                        qty = max(int(max_qty * trade_ml_prob), MIN_TRADE_SIZE)
                    elif TRADE_SIZE_SCALING:
                        qty = max(MIN_TRADE_SIZE, qty)
                    if qty > 0 and qty <= max_qty:
                        self._enter_trade(curr, qty)
            elif self.position > 0:
                self._manage_trade(curr)

        if self.position > 0:
            pnl = (self.m15_df.iloc[-1]['close'] - self.entry_price) * self.position
            self._exit_trade(self.m15_df.iloc[-1]['close'], pnl, self.m15_df.index[-1], "EOD Exit")
        print(f"Trades taken: {len(self.trades)//2} | Final cash: {self.cash:.2f} | Mode: {self.mode}")
        return self.trades

    def _enter_trade(self, curr, qty):
        self.position = qty
        self.entry_price = curr['close']
        self.cash -= qty * self.entry_price
        self.highest_price = self.entry_price
        self.trades.append({'type': 'BUY', 'date': curr.name, 'price': self.entry_price, 'qty': qty})
        print(f"Entered trade: {curr.name} Qty: {qty} Price: {self.entry_price}")

    def _manage_trade(self, curr):
        price = curr['close']
        self.highest_price = max(self.highest_price, price)
        sl = self.entry_price * 0.99
        tp = self.entry_price * 1.02
        trailing_sl = self.highest_price * 0.99
        chand_exit = curr['chandelier_exit']
        stop_loss = max(sl, trailing_sl, chand_exit)
        if price <= stop_loss:
            pnl = (price - self.entry_price) * self.position
            self._exit_trade(price, pnl, curr.name, "Stop/Trail/ChandExit")
        elif price >= tp:
            pnl = (price - self.entry_price) * self.position
            self._exit_trade(price, pnl, curr.name, "Profit Target")

    def _exit_trade(self, price, pnl, date, reason):
        self.cash += self.position * price
        self.trades.append({'type': 'SELL', 'date': date, 'price': price, 'qty': self.position, 'pnl': pnl, 'exit_reason': reason})
        print(f"Exited trade: {date} Qty: {self.position} Price: {price} PnL: {pnl:.2f} Reason: {reason}")
        self.position = 0
        self.entry_price = 0
        self.highest_price = 0

def extract_trade_stats(trades):
    df = pd.DataFrame(trades)
    if df.empty or 'type' not in df.columns or 'pnl' not in df.columns:
        return {}
    closed = df[df['type'] == 'SELL']
    if closed.empty:
        return {}
    stats = {
        'Total Trades': len(closed),
        'Winning Trades': (closed['pnl'] > 0).sum(),
        'Losing Trades': (closed['pnl'] <= 0).sum(),
        'Win Rate %': round((closed['pnl'] > 0).mean() * 100, 2),
        'Avg PnL per Trade': round(closed['pnl'].mean(), 2),
        'Best PnL': round(closed['pnl'].max(), 2),
        'Worst PnL': round(closed['pnl'].min(), 2),
        'Total PnL': round(closed['pnl'].sum(), 2)
    }
    return stats

if __name__ == "__main__":
    def get_symbols_from_data():
        daily_files = os.listdir(DATA_PATHS['daily'])
        return [os.path.splitext(f)[0] for f in daily_files if f.endswith('.csv')]
    symbols = get_symbols_from_data()
    all_stats = []
    for symbol in symbols:
        daily, hourly, m15 = prepare_data_2025(symbol)
        if m15.empty or len(m15) < 30:
            continue
        # Baseline full ML strategy
        ml_model, ml_metrics, ml_index, ml_proba = ml_trade_filter(m15, hourly)
        bt_ml = Backtest2025Next(daily, hourly, m15, ml_model, ml_proba, ml_index, mode='ml_all')
        trades_ml = bt_ml.run()
        stats_ml = extract_trade_stats(trades_ml)
        stats_ml.update({'Symbol': symbol, 'Strategy': 'ML+All Features',
                         'ML Accuracy': round(ml_metrics['accuracy'], 4),
                         'ML Precision': round(ml_metrics['precision'], 4),
                         'ML Recall': round(ml_metrics['recall'], 4)})
        all_stats.append(stats_ml)
        # Breakout only
        bt_bo = Backtest2025Next(daily, hourly, m15, mode='breakout')
        trades_bo = bt_bo.run()
        stats_bo = extract_trade_stats(trades_bo)
        stats_bo.update({'Symbol': symbol, 'Strategy': 'Breakout Only'})
        all_stats.append(stats_bo)
        # Pullback only
        bt_pb = Backtest2025Next(daily, hourly, m15, mode='pullback')
        trades_pb = bt_pb.run()
        stats_pb = extract_trade_stats(trades_pb)
        stats_pb.update({'Symbol': symbol, 'Strategy': 'Pullback Only'})
        all_stats.append(stats_pb)

    df = pd.DataFrame(all_stats)
    df.to_csv("2025_backtest_multi_strategy_summary.csv", index=False)
    print(df)
