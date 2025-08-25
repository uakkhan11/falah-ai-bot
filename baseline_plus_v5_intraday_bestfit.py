import os
import pandas as pd
import numpy as np
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}

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
    df['bb_upper'] = upperband
    df['bb_middle'] = middleband
    df['bb_lower'] = lowerband
    df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['obv'] = talib.OBV(close, volume)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def load_and_filter_data(symbol, years=5):
    cutoff_date = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365*years)
    daily_df = pd.read_csv(os.path.join(DATA_PATHS['daily'], f"{symbol}.csv"), parse_dates=['date'])
    hourly_df = pd.read_csv(os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv"), parse_dates=['date'])
    m15_df = pd.read_csv(os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"), parse_dates=['date'])
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    hourly_df['date'] = pd.to_datetime(hourly_df['date'])
    m15_df['date'] = pd.to_datetime(m15_df['date'])
    daily_df = daily_df[daily_df['date'] >= cutoff_date].reset_index(drop=True)
    hourly_df = hourly_df[hourly_df['date'] >= cutoff_date].reset_index(drop=True)
    m15_df = m15_df[m15_df['date'] >= cutoff_date].reset_index(drop=True)
    return daily_df, hourly_df, m15_df

def prepare_data_for_symbol(symbol, years=5):
    daily_df, hourly_df, m15_df = load_and_filter_data(symbol, years)
    daily_df = compute_indicators(daily_df)
    hourly_df = compute_indicators(hourly_df)
    m15_df = compute_indicators(m15_df)
    daily_df.dropna(subset=['ema8','ema20'], inplace=True)
    hourly_df.dropna(subset=['ema8','ema20'], inplace=True)
    m15_df.dropna(subset=['ema8','ema20'], inplace=True)
    return daily_df, hourly_df, m15_df

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
        daily_close_last = self.daily_df['close'].iloc[-1]
        if daily_close_last <= daily_ema200[-1]:
            return []

        last_hourly = self.hourly_df.iloc[-1]
        if not (last_hourly['ema8'] > last_hourly['ema20'] and last_hourly['rsi_14'] > 50):
            return []

        df = self.df_15m
        for i in range(1,len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            if pd.isna(prev['ema8']) or pd.isna(prev['ema20']): continue
            entry_signal = prev['ema8'] <= prev['ema20'] and curr['ema8'] > curr['ema20']

            if self.position == 0 and entry_signal:
                qty = int(self.cash / curr['close'])
                if qty > 0:
                    self.entry_price = curr['close']
                    self.position = qty
                    self.entry_date = curr['date']
                    self.cash -= qty * self.entry_price
                    self.trades.append({'type':'BUY','date':curr['date'],'price':self.entry_price,'qty':qty})
            elif self.position > 0:
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

        if self.position > 0:
            last_price = df.iloc[-1]['close']
            pnl = (last_price - self.entry_price) * self.position
            self.exit_reasons['EOD Exit'] += 1
            self._exit_trade(last_price, pnl, df.iloc[-1]['date'], "EOD Exit")

        return self.trades

    def _exit_trade(self, price, pnl, date, reason):
        self.cash += self.position * price
        trade_duration = (pd.to_datetime(date) - pd.to_datetime(self.entry_date)).days + 1
        self.trades.append({'type':'SELL', 'date':date, 'price':price, 'qty':self.position,
                            'pnl':pnl, 'duration':trade_duration, 'exit_reason':reason})
        self.position = 0
        self.entry_price = 0
        self.entry_date = None

def create_ml_features(m15_df, hourly_df):
    hourly_df = hourly_df.set_index('date')
    m15_df = m15_df.set_index('date')
    for col in ['ema8', 'ema20', 'rsi_14', 'adx']:
        if col in hourly_df.columns:
            m15_df['hour_' + col] = hourly_df[col].reindex(m15_df.index, method='ffill')
    m15_df['future_return'] = m15_df['close'].shift(-10) / m15_df['close'] - 1
    m15_df['label'] = (m15_df['future_return'] > 0.01).astype(int)
    m15_df.dropna(inplace=True)
    features = m15_df.drop(columns=['label', 'future_return'], errors='ignore')
    labels = m15_df['label']
    non_feature_cols = ['date', 'open', 'high', 'low', 'volume']
    features = features.drop(columns=[col for col in non_feature_cols if col in features.columns], errors='ignore')
    return features, labels

def train_xgboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    importance = model.get_booster().get_score(importance_type='weight')
    importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    return model, accuracy, precision, recall, importance

def generate_report_symbol(trades, symbol, initial_capital):
    df = pd.DataFrame(trades)
    if df.empty:
        return None
    total_trades = len(df) // 2
    wins = df[df['pnl'] > 0]['pnl'].count()
    losses = df[df['pnl'] <= 0]['pnl'].count()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    gross_profit = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = -df[df['pnl'] <= 0]['pnl'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    expectancy = df['pnl'].mean()
    best_trade = df['pnl'].max()
    worst_trade = df['pnl'].min()
    avg_duration = df['duration'].mean()
    total_return = (df['pnl'].sum() / initial_capital) * 100
    return {
        'Symbol': symbol, 'Total Trades': total_trades, 'Wins': wins, 'Losses': losses, 'Win Rate %': win_rate,
        'Profit Factor': profit_factor, 'Expectancy': expectancy, 'Best Trade PnL': best_trade, 'Worst Trade PnL': worst_trade,
        'Avg Trade Duration (days)': avg_duration, 'Total Return %': total_return
    }

def full_grid_search(symbols, indicator_combos, stop_loss_params, profit_target_params, initial_capital=100000):
    results = []
    for indicators in indicator_combos:
        print(f"\nEvaluating indicators combo: {indicators}")
        for sl in stop_loss_params:
            for pt in profit_target_params:
                print(f"Stop Loss: {sl}, Profit Target: {pt}")
                all_trades = []
                ml_accuracies, ml_precisions, ml_recalls = [], [], []
                feature_importances_accum = {}
                for symbol in symbols:
                    try:
                        daily_df, hourly_df, m15_df = prepare_data_for_symbol(symbol)
                        if len(daily_df) < 100 or len(hourly_df) < 100 or len(m15_df) < 100:
                            print(f"Skipping {symbol} due to insufficient data.")
                            continue
                        if 'daily' not in indicators:
                            daily_df = daily_df.drop(columns=[col for col in daily_df.columns if col.startswith(tuple(['ema','rsi','macd','adx','atr','sar','roc','cmo','bb_','adosc','obv']))], errors='ignore')
                        if 'hourly' not in indicators:
                            hourly_df = hourly_df.drop(columns=[col for col in hourly_df.columns if col.startswith(tuple(['ema','rsi','macd','adx','atr','sar','roc','cmo','bb_','adosc','obv']))], errors='ignore')
                        if '15minute' not in indicators:
                            m15_df = m15_df.drop(columns=[col for col in m15_df.columns if col.startswith(tuple(['ema','rsi','macd','adx','atr','sar','roc','cmo','bb_','adosc','obv']))], errors='ignore')
                        strategy = BacktestStrategy(
                            daily_df, hourly_df, m15_df,
                            initial_capital,
                            params={'stop_loss_pct': sl, 'profit_target_pct': pt}
                        )
                        trades = strategy.run_backtest()
                        all_trades.extend(trades)
                        X, y = create_ml_features(m15_df, hourly_df)
                        model, acc, prec, rec, feat_imp = train_xgboost_model(X, y)
                        ml_accuracies.append(acc)
                        ml_precisions.append(prec)
                        ml_recalls.append(rec)
                        for k,v in feat_imp.items():
                            feature_importances_accum[k] = feature_importances_accum.get(k,0) + v
                    except Exception as e:
                        print(f"Error with {symbol}: {e}")
                if not all_trades:
                    print("No trades executed for this setting.")
                    continue
                backtest_metrics = generate_report_symbol(all_trades, 'ALL_SYMBOLS', initial_capital)
                avg_acc = np.mean(ml_accuracies) if ml_accuracies else None
                avg_prec = np.mean(ml_precisions) if ml_precisions else None
                avg_rec = np.mean(ml_recalls) if ml_recalls else None
                sorted_feat_imp = dict(sorted(feature_importances_accum.items(), key=lambda item: item[1], reverse=True))
                result = {
                    'Indicators': ','.join(indicators),
                    'Stop Loss %': sl,
                    'Profit Target %': pt,
                    'Total Trades': backtest_metrics['Total Trades'] if backtest_metrics else 0,
                    'Win Rate %': backtest_metrics['Win Rate %'] if backtest_metrics else 0,
                    'Profit Factor': backtest_metrics['Profit Factor'] if backtest_metrics else 0,
                    'Total Return %': backtest_metrics['Total Return %'] if backtest_metrics else 0,
                    'ML Accuracy': avg_acc,
                    'ML Precision': avg_prec,
                    'ML Recall': avg_rec,
                    'Top Features': list(sorted_feat_imp.keys())[:10]
                }
                results.append(result)
                print(f"Grid search result: {result}")
    return pd.DataFrame(results)

if __name__ == "__main__":
    symbols = get_symbols_from_daily_data()
    indicator_combos = [
        {'daily', 'hourly', '15minute'},
        {'daily', 'hourly'},
        {'daily', '15minute'},
        {'hourly', '15minute'},
        {'daily'},
        {'hourly'},
        {'15minute'},
    ]
    stop_loss_params = [0.0025, 0.005, 0.0075]
    profit_target_params = [0.01, 0.015, 0.02]
    results_df = full_grid_search(symbols, indicator_combos, stop_loss_params, profit_target_params)
    results_df.to_csv("full_grid_search_results.csv", index=False)
    print("\nGrid search complete, results saved to 'full_grid_search_results.csv'.")
