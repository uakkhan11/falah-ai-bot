import os
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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

# CONFIGURABLE PARAMETERS
ML_PROBA_THRESHOLD = 0.5       # Lower to increase trade frequency
MIN_TRADE_SIZE = 1
TRADE_SIZE_SCALING = True

#############################
# Data Load: Year 2025 only
#############################
def load_and_filter_2025(symbol):
    def filter_year(df):
        df['date'] = pd.to_datetime(df['date'])
        return df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)
    daily = pd.read_csv(os.path.join(DATA_PATHS['daily'], f"{symbol}.csv"))
    hourly = pd.read_csv(os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv"))
    m15 = pd.read_csv(os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"))
    daily, hourly, m15 = filter_year(daily), filter_year(hourly), filter_year(m15)
    return daily, hourly, m15

def add_ichimoku(df):
    high = df['high']
    low = df['low']
    close = df['close']

    period9_high = high.rolling(window=9).max()
    period9_low = low.rolling(window=9).min()
    df['tenkan_sen'] = (period9_high + period9_low) / 2

    period26_high = high.rolling(window=26).max()
    period26_low = low.rolling(window=26).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2

    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    period52_high = high.rolling(window=52).max()
    period52_low = low.rolling(window=52).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

    df['chikou_span'] = close.shift(-26)

    df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b'])

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    df = add_ichimoku(df)
    
    return df

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

    df = add_ichimoku(df)  # Correctly call here once, not inside add_ichimoku itself

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    df = add_ichimoku(df)

    return df

def prepare_data_2025(symbol):
    daily, hourly, m15 = load_and_filter_2025(symbol)
    daily, hourly, m15 = compute_indicators(daily), compute_indicators(hourly), compute_indicators(m15)
    daily.dropna(subset=['ema8','ema20'], inplace=True)
    hourly.dropna(subset=['ema8','ema20'], inplace=True)
    m15.dropna(subset=['ema8','ema20'], inplace=True)
    return daily, hourly, m15

#####################
# ML filter - only best features
#####################
FEATURES = ["adx", "atr", "volume_ratio", "adosc", "hour_adx", "volume_sma", "macd_hist", "vwap", "roc", "obv"]

def ml_trade_filter_tuned(m15_df, hourly_df):
    hourly_df = hourly_df.set_index('date')
    m15_df = m15_df.set_index('date')
    for col in set(['adx', 'atr', 'macd_hist']) & set(hourly_df.columns):
        m15_df['hour_' + col] = hourly_df[col].reindex(m15_df.index, method='ffill')
    m15_df['future_return'] = m15_df['close'].shift(-10) / m15_df['close'] - 1
    m15_df['label'] = (m15_df['future_return'] > 0.01).astype(int)
    m15_df.dropna(subset=FEATURES + ['label'], inplace=True)
    X = m15_df[FEATURES]
    y = m15_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
        'scale_pos_weight': [1, 5, 10],  # handle imbalance by weighting positive class
        'learning_rate': [0.01, 0.1]
    }

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    grid = GridSearchCV(model, param_grid, scoring='recall', cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'clf_report': classification_report(y_test, y_pred, output_dict=True),
        'best_params': grid.best_params_
    }
    proba_all = best_model.predict_proba(X)[:, 1]
    return best_model, metrics, m15_df.index, proba_all

#################
# Core: "Live-style" backtest for 2025 with ML trade confidence scaling
#################
class Backtest2025Next:
    def __init__(self, daily_df, hourly_df, m15_df, ml_model, ml_proba, ml_index, init_cap=100000):
        self.daily_df = daily_df
        self.hourly_df = hourly_df
        self.m15_df = m15_df
        self.ml_model = ml_model
        self.ml_proba = pd.Series(ml_proba, index=ml_index)
        self.cash = init_cap
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.highest_price = 0

    def run(self):
        if self.m15_df.empty:
            return []

        for i in range(1, len(self.m15_df)):
            prev = self.m15_df.iloc[i-1]
            curr = self.m15_df.iloc[i]
            curr_idx = curr.name

            trade_ml_prob = self.ml_proba.get(curr_idx, 0)

            # Relaxed timeframe confirmation: only daily & hourly EMA check
            daily_ok = (self.daily_df['ema8'].iloc[-1] > self.daily_df['ema20'].iloc[-1])
            hourly_ok = (self.hourly_df['ema8'].iloc[-1] > self.hourly_df['ema20'].iloc[-1])

            if self.position == 0:
                if daily_ok and hourly_ok and curr['adx'] > 20 and trade_ml_prob > ML_PROBA_THRESHOLD:
                    max_qty = int(self.cash / curr['close'])
                    qty = MIN_TRADE_SIZE
                    if TRADE_SIZE_SCALING:
                        qty = max(int(max_qty * trade_ml_prob), MIN_TRADE_SIZE)
                    if qty > 0 and qty <= max_qty:
                        self._enter_trade(curr, qty)
            elif self.position > 0:
                self._manage_trade(curr)

        if self.position > 0:
            pnl = (self.m15_df.iloc[-1]['close'] - self.entry_price) * self.position
            self._exit_trade(self.m15_df.iloc[-1]['close'], pnl, self.m15_df.index[-1], "EOD Exit")

        print(f"Trades taken: {len(self.trades)//2} | Final cash: {self.cash:.2f}")
        return self.trades

    def _enter_trade(self, curr, qty):
        in_cloud = (curr['close'] > min(curr['senkou_span_a'], curr['senkou_span_b']) and
                    curr['close'] < max(curr['senkou_span_a'], curr['senkou_span_b']))
        tenkan_above_kijun = curr['tenkan_sen'] > curr['kijun_sen']
        self.position = qty
        self.entry_price = curr['close']
        self.cash -= qty * self.entry_price
        self.highest_price = self.entry_price
        self.trades.append({
            'type': 'BUY',
            'date': curr.name,
            'price': self.entry_price,
            'qty': qty,
            'in_cloud': in_cloud,
            'tenkan_above_kijun': tenkan_above_kijun
        })
        print(f"Entered trade: {curr.name} Qty: {qty} Price: {self.entry_price} InCloud: {in_cloud} TenkanAboveKijun: {tenkan_above_kijun}")

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

def analyze_ichimoku_trades(trades_df):
    if trades_df.empty:
        return "No trades to analyze.\n"

    trades_df['win'] = trades_df['pnl'] > 0
    report = ""

    if 'in_cloud' in trades_df.columns:
        win_rate_cloud = trades_df.groupby('in_cloud')['win'].mean()
        report += "Win Rate by Ichimoku Cloud Presence:\n"
        for val, rate in win_rate_cloud.items():
            report += f"  In Cloud: {val} -> Win Rate: {rate:.2%}\n"

        sns.barplot(x=win_rate_cloud.index.astype(str), y=win_rate_cloud.values)
        plt.title("Win Rate by Ichimoku Cloud Presence")
        plt.ylabel("Win Rate")
        plt.xlabel("Trade Entry Inside Cloud")
        plt.savefig("win_rate_by_ichimoku_cloud.png")
        plt.close()

    if 'tenkan_above_kijun' in trades_df.columns:
        win_rate_tk = trades_df.groupby('tenkan_above_kijun')['win'].mean()
        report += "Win Rate by Tenkan > Kijun:\n"
        for val, rate in win_rate_tk.items():
            report += f"  Tenkan above Kijun: {val} -> Win Rate: {rate:.2%}\n"

        sns.barplot(x=win_rate_tk.index.astype(str), y=win_rate_tk.values)
        plt.title("Win Rate by Tenkan > Kijun")
        plt.ylabel("Win Rate")
        plt.xlabel("Tenkan Above Kijun at Entry")
        plt.savefig("win_rate_by_tenkan_kijun.png")
        plt.close()

    report += "\nTrade PnL distribution saved as histogram plot.\n"

    sns.histplot(data=trades_df, x="pnl", bins=30, kde=True)
    plt.title("Distribution of Trade PnL")
    plt.savefig("trade_pnl_distribution.png")
    plt.close()

    return report


if __name__ == "__main__":
    def get_symbols_from_data():
        daily_files = os.listdir(DATA_PATHS['daily'])
        return [os.path.splitext(f)[0] for f in daily_files if f.endswith('.csv')]

    symbols = get_symbols_from_data()
    all_stats = []
    report_lines = []
    all_trades = []  # Collect trades from all symbols

    for symbol in symbols:
        daily, hourly, m15 = prepare_data_2025(symbol)
        if m15.empty or len(m15) < 30:
            continue
        ml_model, ml_metrics, ml_index, ml_proba = ml_trade_filter_tuned(m15, hourly)
        bt = Backtest2025Next(daily, hourly, m15, ml_model, ml_proba, ml_index)
        trades = bt.run()
        # Collect trades for later analysis
        all_trades.extend(trades)
        stats = extract_trade_stats(trades)
        stats['Symbol'] = symbol
        stats['ML Accuracy'] = round(ml_metrics['accuracy'], 4)
        stats['ML Precision'] = round(ml_metrics['precision'], 4)
        stats['ML Recall'] = round(ml_metrics['recall'], 4)
        all_stats.append(stats)
        report_lines.append(
            f"\n=== {symbol} ===\n"
            + "\n".join([f"{k}: {v}" for k, v in stats.items()])
            + f"\nML Classification Report:\n{ml_metrics['clf_report']}\n"
        )

    df = pd.DataFrame(all_stats)
df.to_csv("2025_backtest_next_summary.csv", index=False)

all_trades_df = pd.DataFrame(all_trades)  # Create DataFrame after collecting all trades

with open("2025_detailed_next_report.txt", "a") as f:  # Append Ichimoku summary text
    ichimoku_report = analyze_ichimoku_trades(all_trades_df)
    f.write("\n=== Ichimoku Indicator Trade Analysis ===\n")
    f.write(ichimoku_report)

print(df)
print("\nNext phase backtest complete. Summary saved.")

