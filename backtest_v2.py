import os
import pandas as pd
import numpy as np
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import warnings
import json

warnings.filterwarnings("ignore")

# ================================
# Paths and Constants
# ================================
BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}

# ================================
# Data Loading & Indicator Computation
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

    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)

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
# Backtesting Strategy Class
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
# ML Feature Preparation and Training
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
    features = m15_df.drop(columns=['label', 'future_return', 'open', 'high', 'low', 'volume'], errors='ignore')
    return features, labels

def train_xgboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    importance = model.get_booster().get_score(importance_type='weight')
    importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    return model, accuracy, precision, recall, importance, clf_report

# ================================
# Reporting Functions
# ================================
def extract_trade_stats(trades):
    df = pd.DataFrame(trades)
    if df.empty or 'pnl' not in df.columns:
        return {}

    df_closed = df[df['type'] == 'SELL']
    stats = {
        'Total Trades': len(df_closed),
        'Winning Trades': (df_closed['pnl'] > 0).sum(),
        'Losing Trades': (df_closed['pnl'] <= 0).sum(),
        'Win Rate %': round((df_closed['pnl'] > 0).mean() * 100, 2),
        'Avg PnL per Trade': round(df_closed['pnl'].mean(), 4),
        'Best PnL': round(df_closed['pnl'].max(), 4),
        'Worst PnL': round(df_closed['pnl'].min(), 4),
        'Avg Trade Duration (days)': round(df_closed['duration'].mean(), 2),
        'Stop Loss Hits': sum(t['exit_reason']=='Stop Loss' for t in trades if 'exit_reason' in t),
        'Profit Target Hits': sum(t['exit_reason']=='Profit Target' for t in trades if 'exit_reason' in t),
        'EOD Exits': sum(t['exit_reason']=='EOD Exit' for t in trades if 'exit_reason' in t),
        'Total PnL': round(df_closed['pnl'].sum(), 4)
    }
    return stats

def format_classification_report(report_dict):
    lines = []
    for label in ['0', '1', 'accuracy', 'macro avg', 'weighted avg']:
        if label in report_dict:
            if label in ['accuracy']:
                lines.append(f"Accuracy: {report_dict[label]:.4f}")
            else:
                rec = report_dict[label].get('recall', np.nan)
                prec = report_dict[label].get('precision', np.nan)
                f1 = report_dict[label].get('f1-score', np.nan)
                support = report_dict[label].get('support', np.nan)
                lines.append(f"{label} -> Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}, Support: {support}")
    return "\n".join(lines)

def save_detailed_report(filename, title, results_dict, clf_report, trade_stats, ml_metrics, feature_imp):
    with open(filename, "a") as f:
        f.write("="*80 + "\n")
        f.write(f"{title}\n")
        f.write("="*80 + "\n")
        f.write("Backtest Trade Stats:\n")
        for k,v in trade_stats.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nML Classification Report:\n")
        f.write(format_classification_report(clf_report))
        f.write("\n\nML Metrics:\n")
        for k,v in ml_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTop Feature Importances:\n")
        for feat, imp in feature_imp.items():
            f.write(f"  {feat}: {imp}\n")
        f.write("\n\n")

# ================================
# Full Grid Search with Detailed Reports
# ================================
def full_grid_search(symbols, indicator_combos, stop_loss_params, profit_target_params, initial_capital=100000):
    results = []
    # Clear or create report file
    report_file = "detailed_summary_report.txt"
    with open(report_file, "w") as f:
        f.write("Detailed Backtest and ML Report\n\n")

    for indicators in indicator_combos:
        indicator_str = ",".join(sorted(indicators))
        for sl in stop_loss_params:
            for pt in profit_target_params:
                all_trades, ml_accuracies, ml_precisions, ml_recalls = [], [], [], []
                feature_importances_accum = {}
                trade_stats_per_symbol = {}
                ml_reports_per_symbol = {}
                for symbol in symbols:
                    try:
                        daily_df, hourly_df, m15_df = prepare_data_for_symbol(symbol)
                        if len(daily_df)<100 or len(hourly_df)<100 or len(m15_df)<100:
                            continue
                        # Drop unwanted indicator columns based on current combo
                        if 'daily' not in indicators:
                            daily_df = daily_df.drop(columns=[col for col in daily_df.columns if any(col.startswith(p) for p in ['ema','rsi','macd','adx','atr','sar','roc','cmo','bb','adosc','obv'])], errors='ignore')
                        if 'hourly' not in indicators:
                            hourly_df = hourly_df.drop(columns=[col for col in hourly_df.columns if any(col.startswith(p) for p in ['ema','rsi','macd','adx','atr','sar','roc','cmo','bb','adosc','obv'])], errors='ignore')
                        if '15minute' not in indicators:
                            m15_df = m15_df.drop(columns=[col for col in m15_df.columns if any(col.startswith(p) for p in ['ema','rsi','macd','adx','atr','sar','roc','cmo','bb','adosc','obv'])], errors='ignore')

                        strategy = BacktestStrategy(daily_df, hourly_df, m15_df, initial_capital, params={'stop_loss_pct':sl, 'profit_target_pct':pt})
                        trades = strategy.run_backtest()
                        all_trades.extend(trades)

                        X, y = create_ml_features(m15_df, hourly_df)
                        model, acc, prec, rec, feat_imp, clf_report = train_xgboost_model(X, y)

                        ml_accuracies.append(acc)
                        ml_precisions.append(prec)
                        ml_recalls.append(rec)
                        
                        for k,v in feat_imp.items():
                            feature_importances_accum[k] = feature_importances_accum.get(k,0)+v

                        # Detailed per-symbol stats
                        trade_stats_per_symbol[symbol] = extract_trade_stats(trades)
                        ml_reports_per_symbol[symbol] = clf_report

                    except Exception as e:
                        print(f"Error processing {symbol}: {e}")

                if not all_trades:
                    continue

                backtest_metrics = extract_trade_stats(all_trades)
                avg_acc = np.mean(ml_accuracies) if ml_accuracies else 0
                avg_prec = np.mean(ml_precisions) if ml_precisions else 0
                avg_rec = np.mean(ml_recalls) if ml_recalls else 0
                ml_metrics = {'Accuracy': avg_acc, 'Precision': avg_prec, 'Recall': avg_rec}

                sorted_feat_imp = dict(sorted(feature_importances_accum.items(), key=lambda item: item[1], reverse=True))
                result_row = {
                    'Indicators': indicator_str,
                    'Stop Loss %': sl,
                    'Profit Target %': pt,
                    'Total Trades': backtest_metrics.get('Total Trades', 0),
                    'Win Rate %': backtest_metrics.get('Win Rate %', 0),
                    'Profit Factor': (np.sum([t['pnl'] for t in all_trades if 'pnl' in t and t['pnl'] > 0]) /
                                      -np.sum([t['pnl'] for t in all_trades if 'pnl' in t and t['pnl'] <= 0]) 
                                      if np.sum([t['pnl'] for t in all_trades if 'pnl' in t and t['pnl'] <= 0]) < 0 else np.inf),
                    'Total Return %': (np.sum([t['pnl'] for t in all_trades if 'pnl' in t]) / initial_capital * 100),
                    'ML Accuracy': round(avg_acc,4),
                    'ML Precision': round(avg_prec,4),
                    'ML Recall': round(avg_rec,4),
                    'Top Features': ", ".join(list(sorted_feat_imp.keys())[:10])
                }
                results.append(result_row)
                print(f"Completed: Indicators={indicator_str}, SL={sl}, PT={pt}, Trades={result_row['Total Trades']}, Return={result_row['Total Return %']:.2f}%")

                # Save detailed report per config
                title = (f"Indicator Set: {indicator_str}\n"
                         f"Stop Loss: {sl}, Profit Target: {pt}\n")
                with open(report_file, "a") as f:
                    f.write("\n" + "="*80 + "\n")
                    f.write(title)
                    f.write("="*80 + "\n")
                    f.write("Overall Backtest Stats:\n")
                    for k, v in backtest_metrics.items():
                        f.write(f"{k}: {v}\n")
                    f.write("\nML Metrics (average over symbols):\n")
                    for k, v in ml_metrics.items():
                        f.write(f"{k}: {v:.4f}\n")
                    f.write("\nTop Feature Importances:\n")
                    for feat, imp in sorted_feat_imp.items():
                        f.write(f"{feat}: {imp}\n")

                    f.write("\nPer-Symbol Trade Stats:\n")
                    for sym, stats in trade_stats_per_symbol.items():
                        f.write(f"Symbol: {sym}\n")
                        for k, v in stats.items():
                            f.write(f"  {k}: {v}\n")
                        f.write("\n")

                    f.write("\n")  # spacer for next config already appended

    # Final summary CSV for all results
    results_df = pd.DataFrame(results)
    results_df.to_csv("full_grid_search_results.csv", index=False)

    # Also print a concise summary
    print("\nAll runs completed. Summary saved to 'full_grid_search_results.csv' and detailed text report.")
    print_consolidated_report(results_df, save_path="summary_consolidated_report.txt")

    return results_df

# ================================
# Consolidated Summary & Export
# ================================
def print_consolidated_report(results_df, save_path="summary_consolidated_report.txt"):
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
        lines.append(f"- Highest Return %: {best_return['Total Return %']:.2f} | Indicators={best_return['Indicators']} | SL={best_return['Stop Loss %']} | PT={best_return['Profit Target %']}")
        lines.append(f"- Highest Win Rate %: {best_winrate['Win Rate %']:.2f} | Indicators={best_winrate['Indicators']} | SL={best_winrate['Stop Loss %']} | PT={best_winrate['Profit Target %']}")
        lines.append(f"- Highest Profit Factor: {best_profitfactor['Profit Factor']:.2f} | Indicators={best_profitfactor['Indicators']} | SL={best_profitfactor['Stop Loss %']} | PT={best_profitfactor['Profit Target %']}")

        lines.append("\n🤖 ML Metrics (Average across all runs):")
        lines.append(f"- Accuracy: {results_df['ML Accuracy'].mean():.4f}")
        lines.append(f"- Precision: {results_df['ML Precision'].mean():.4f}")
        lines.append(f"- Recall: {results_df['ML Recall'].mean():.4f}")

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

    print(f"\n📝 Consolidated report saved to '{save_path}'")

# ================================
# Main Entry Point
# ================================
if __name__ == "__main__":
    symbols = get_symbols_from_daily_data()
    indicator_combos = [
        {'daily','hourly','15minute'},{'daily','hourly'},{'daily','15minute'},
        {'hourly','15minute'},{'daily'},{'hourly'},{'15minute'}
    ]
    stop_loss_params = [0.005, 0.0075, 0.01]  # Finer grid around best SL
    profit_target_params = [0.0125, 0.015, 0.02]  # Finer grid around best PT

    results_df = full_grid_search(symbols, indicator_combos, stop_loss_params, profit_target_params)
