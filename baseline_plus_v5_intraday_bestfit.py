import os
import pandas as pd
import numpy as np
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")

# ================================
# Paths
# ================================
BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}

# ================================
# Data Preparation
# ================================
def get_symbols_from_daily_data():
    daily_files = os.listdir(DATA_PATHS['daily'])
    return [os.path.splitext(f)[0] for f in daily_files if f.endswith('.csv')]

def compute_indicators(df):
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill()
    df['high'] = pd.to_numeric(df['high'], errors='coerce').ffill()
    df['low'] = pd.to_numeric(df['low'], errors='coerce').ffill()
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    close, high, low, volume = df['close'].values, df['high'].values, df['low'].values, df['volume'].values

    df['ema8'] = talib.EMA(close, timeperiod=8)
    df['ema20'] = talib.EMA(close, timeperiod=20)
    df['rsi_14'] = talib.RSI(close, timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    df['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
    df['roc'] = talib.ROC(close, timeperiod=10)
    df['cmo'] = talib.CMO(close, timeperiod=14)
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = upperband, middleband, lowerband
    df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['obv'] = talib.OBV(close, volume)

    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def load_and_filter_data(symbol, years=5):
    cutoff_date = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365 * years)
    daily_df = pd.read_csv(os.path.join(DATA_PATHS['daily'], f"{symbol}.csv"), parse_dates=['date'])
    hourly_df = pd.read_csv(os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv"), parse_dates=['date'])
    m15_df = pd.read_csv(os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"), parse_dates=['date'])

    for df in [daily_df, hourly_df, m15_df]:
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= cutoff_date].reset_index(drop=True)
    return daily_df, hourly_df, m15_df

def prepare_data_for_symbol(symbol, years=5):
    daily_df, hourly_df, m15_df = load_and_filter_data(symbol, years)
    daily_df, hourly_df, m15_df = compute_indicators(daily_df), compute_indicators(hourly_df), compute_indicators(m15_df)
    daily_df.dropna(subset=['ema8','ema20'], inplace=True)
    hourly_df.dropna(subset=['ema8','ema20'], inplace=True)
    m15_df.dropna(subset=['ema8','ema20'], inplace=True)
    return daily_df, hourly_df, m15_df

# ================================
# Backtesting Strategy
# ================================
class BacktestStrategy:
    def __init__(self, daily_df, hourly_df, m15_df, initial_capital=100000, params=None):
        self.daily_df = daily_df.sort_values('date').reset_index(drop=True)
        self.hourly_df = hourly_df.sort_values('date').reset_index(drop=True)
        self.df_15m = m15_df.sort_values('date').reset_index(drop=True)
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.entry_price = 0
        self.entry_date = None
        self.trades = []
        self.exit_reasons = {'StopLoss':0, 'ProfitTarget':0, 'EOD Exit':0}
        self.params = params or {'stop_loss_pct':0.0025, 'profit_target_pct':0.01}

    def run_backtest(self):
        daily_ema200 = talib.EMA(self.daily_df['close'].values, timeperiod=200)
        if self.daily_df['close'].iloc[-1] <= daily_ema200[-1]:
            return []

        last_hourly = self.hourly_df.iloc[-1]
        if not (last_hourly['ema8'] > last_hourly['ema20'] and last_hourly['rsi_14'] > 50):
            return []

        df = self.df_15m
        for i in range(1, len(df)):
            prev, curr = df.iloc[i-1], df.iloc[i]
            if pd.isna(prev['ema8']) or pd.isna(prev['ema20']): 
                continue
            entry_signal = prev['ema8'] <= prev['ema20'] and curr['ema8'] > curr['ema20']
            if self.position == 0 and entry_signal:
                qty = int(self.cash / curr['close'])
                if qty > 0:
                    self._enter_trade(curr, qty)
            elif self.position > 0:
                self._manage_trade(curr)

        if self.position > 0:
            last_row = df.iloc[-1]
            pnl = (last_row['close'] - self.entry_price) * self.position
            self.exit_reasons['EOD Exit'] += 1
            self._exit_trade(last_row['close'], pnl, last_row['date'], "EOD Exit")

        return self.trades

    def _enter_trade(self, curr, qty):
        self.entry_price = curr['close']
        self.position = qty
        self.entry_date = curr['date']
        self.cash -= qty * self.entry_price
        self.trades.append({'type':'BUY','date':curr['date'],'price':self.entry_price,'qty':qty})

    def _manage_trade(self, curr):
        current_price = curr['close']
        stop_loss = self.entry_price * (1 - self.params['stop_loss_pct'])
        profit_target = self.entry_price * (1 + self.params['profit_target_pct'])
        if current_price <= stop_loss:
            pnl = (current_price - self.entry_price) * self.position
            self.exit_reasons['StopLoss'] += 1
            self._exit_trade(current_price, pnl, curr['date'], "Stop Loss")
        elif current_price >= profit_target:
            pnl = (current_price - self.entry_price) * self.position
            self.exit_reasons['ProfitTarget'] += 1
            self._exit_trade(current_price, pnl, curr['date'], "Profit Target")

    def _exit_trade(self, price, pnl, date, reason):
        self.cash += self.position * price
        trade_duration = (pd.to_datetime(date) - pd.to_datetime(self.entry_date)).days + 1
        self.trades.append({'type':'SELL', 'date':date, 'price':price, 'qty':self.position,
                            'pnl':pnl, 'duration':trade_duration, 'exit_reason':reason})
        self.position, self.entry_price, self.entry_date = 0, 0, None

# ================================
# ML Features & Training
# ================================
def create_ml_features(m15_df, hourly_df):
    hourly_df = hourly_df.set_index('date')
    m15_df = m15_df.set_index('date')
    for col in ['ema8', 'ema20', 'rsi_14', 'adx']:
        if col in hourly_df.columns:
            m15_df['hour_' + col] = hourly_df[col].reindex(m15_df.index, method='ffill')
    m15_df['future_return'] = m15_df['close'].shift(-10) / m15_df['close'] - 1
    m15_df['label'] = (m15_df['future_return'] > 0.01).astype(int)
    m15_df.dropna(inplace=True)
    labels = m15_df['label']
    features = m15_df.drop(columns=['label', 'future_return', 'open','high','low','volume'], errors='ignore')
    return features, labels

def train_xgboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    importance = model.get_booster().get_score(importance_type='weight')
    importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    return model, accuracy, precision, recall, importance

# ================================
# Reporting
# ================================
def generate_report_symbol(trades, symbol, initial_capital):
    df = pd.DataFrame(trades)
    if df.empty or 'pnl' not in df.columns:
        return None
    df_closed = df[df['type'] == 'SELL']
    if df_closed.empty:
        return None
    total_trades = len(df_closed)
    wins = (df_closed['pnl'] > 0).sum()
    losses = (df_closed['pnl'] <= 0).sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    gross_profit = df_closed[df_closed['pnl'] > 0]['pnl'].sum()
    gross_loss = -df_closed[df_closed['pnl'] <= 0]['pnl'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    return {
        'Symbol': symbol,
        'Total Trades': total_trades,
        'Wins': wins, 'Losses': losses,
        'Win Rate %': round(win_rate,2),
        'Profit Factor': round(profit_factor,2),
        'Expectancy': round(df_closed['pnl'].mean(),2),
        'Best Trade PnL': round(df_closed['pnl'].max(),2),
        'Worst Trade PnL': round(df_closed['pnl'].min(),2),
        'Avg Trade Duration (days)': round(df_closed['duration'].mean(),2),
        'Total Return %': round((df_closed['pnl'].sum() / initial_capital) * 100,2)
    }

# ================================
# Grid Search
# ================================
def full_grid_search(symbols, indicator_combos, stop_loss_params, profit_target_params, initial_capital=100000):
    results = []
    for indicators in indicator_combos:
        print(f"\nEvaluating indicators combo: {indicators}")
        for sl in stop_loss_params:
            for pt in profit_target_params:
                print(f" Stop Loss: {sl}, Profit Target: {pt}")
                all_trades, ml_accuracies, ml_precisions, ml_recalls = [], [], [], []
                feature_importances_accum = {}
                for symbol in symbols:
                    try:
                        daily_df, hourly_df, m15_df = prepare_data_for_symbol(symbol)
                        if len(daily_df)<100 or len(hourly_df)<100 or len(m15_df)<100:
                            print(f"Skipping {symbol}: insufficient data.")
                            continue
                        strategy = BacktestStrategy(daily_df, hourly_df, m15_df, initial_capital, params={'stop_loss_pct':sl, 'profit_target_pct':pt})
                        trades = strategy.run_backtest()
                        all_trades.extend(trades)
                        X, y = create_ml_features(m15_df, hourly_df)
                        model, acc, prec, rec, feat_imp = train_xgboost_model(X, y)
                        ml_accuracies.append(acc); ml_precisions.append(prec); ml_recalls.append(rec)
                        for k,v in feat_imp.items(): feature_importances_accum[k] = feature_importances_accum.get(k,0)+v
                    except Exception as e:
                        print(f"Error with {symbol}: {e}")
                if not all_trades:
                    print(" No trades executed for this setting.")
                    continue
                backtest_metrics = generate_report_symbol(all_trades, 'ALL_SYMBOLS', initial_capital)
                avg_acc, avg_prec, avg_rec = np.mean(ml_accuracies), np.mean(ml_precisions), np.mean(ml_recalls)
                sorted_feat_imp = dict(sorted(feature_importances_accum.items(), key=lambda item: item[1], reverse=True))
                result = {
                    'Indicators': ','.join(indicators),
                    'Stop Loss %': sl, 'Profit Target %': pt,
                    'Total Trades': backtest_metrics['Total Trades'],
                    'Win Rate %': backtest_metrics['Win Rate %'],
                    'Profit Factor': backtest_metrics['Profit Factor'],
                    'Total Return %': backtest_metrics['Total Return %'],
                    'ML Accuracy': round(avg_acc,4), 'ML Precision': round(avg_prec,4), 'ML Recall': round(avg_rec,4),
                    'Top Features': ", ".join(list(sorted_feat_imp.keys())[:5])
                }
                results.append(result)
                print(f" ✅ Result: {result}")
    return pd.DataFrame(results)

# ================================
# Consolidated Report Export + Terminal
# ================================
def print_consolidated_report(results_df, save_path="summary_report.txt"):
    lines = []
    lines.append("="*100)
    lines.append("📊 CONSOLIDATED REPORT")
    lines.append("="*100)
    total_runs = len(results_df)
    lines.append(f"Total Strategies Tested: {total_runs}")
    if total_runs == 0:
        lines.append("No results available.")
    else:
        best_return = results_df.loc[results_df['Total Return %'].idxmax()]
        best_winrate = results_df.loc[results_df['Win Rate %'].idxmax()]
        best_profitfactor = results_df.loc[results_df['Profit Factor'].idxmax()]
        lines.append("\n🏆 Best Strategies:")
        lines.append(f"- Highest Return %: {best_return['Total Return %']} | Indicators={best_return['Indicators']} | SL={best_return['Stop Loss %']} | PT={best_return['Profit Target %']}")
        lines.append(f"- Highest Win Rate %: {best_winrate['Win Rate %']} | Indicators={best_winrate['Indicators']} | SL={best_winrate['Stop Loss %']} | PT={best_winrate['Profit Target %']}")
        lines.append(f"- Highest Profit Factor: {best_profitfactor['Profit Factor']} | Indicators={best_profitfactor['Indicators']} | SL={best_profitfactor['Stop Loss %']} | PT={best_profitfactor['Profit Target %']}")
        lines.append("\n🤖 ML Metrics (Average across all runs)")
        lines.append(f"- Accuracy: {results_df['ML Accuracy'].mean():.3f}")
        lines.append(f"- Precision: {results_df['ML Precision'].mean():.3f}")
        lines.append(f"- Recall: {results_df['ML Recall'].mean():.3f}")
        features_flat = []
        for feat in results_df['Top Features']:
            if isinstance(feat, str):
                features_flat.extend([f.strip() for f in feat.split(",")])
        common_features = pd.Series(features_flat).value_counts().head(10)
        lines.append("\n🔥 Top ML Features (most frequently important):")
        for i, (feat, count) in enumerate(common_features.items(), 1):
            lines.append(f"{i}. {feat} ({count} times)")
        lines.append("\n📈 Performance Grouped by Indicator Combo:")
        group_stats = results_df.groupby("Indicators")[["Total Return %", "Win Rate %", "Profit Factor"]].mean().sort_values("Total Return %", ascending=False)
        lines.append(group_stats.to_string(float_format=lambda x: f"{x:0.2f}"))
    lines.append("="*100)
    report_text = "\n".join(lines)
    print(report_text)
    with open(save_path, "w") as f:
        f.write(report_text)
    print(f"\n📝 Consolidated report also saved to '{save_path}'")

# ================================
# Main
# ================================
if __name__ == "__main__":
    symbols = get_symbols_from_daily_data()
    indicator_combos = [
        {'daily','hourly','15minute'},{'daily','hourly'},{'daily','15minute'},
        {'hourly','15minute'},{'daily'},{'hourly'},{'15minute'}
    ]
    stop_loss_params = [0.0025, 0.005, 0.0075]
    profit_target_params = [0.01, 0.015, 0.02]
    results_df = full_grid_search(symbols, indicator_combos, stop_loss_params, profit_target_params)
    results_df.to_csv("full_grid_search_results.csv", index=False)
    print("\n🚀 Grid search complete, results saved to 'full_grid_search_results.csv'.")
    print_consolidated_report(results_df, save_path="summary_report.txt")
