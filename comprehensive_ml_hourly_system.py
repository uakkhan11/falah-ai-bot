
import os
import pandas as pd
import numpy as np
import ta
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
warnings.filterwarnings("ignore")

# =============================================================================
# COMPREHENSIVE ML TRADING SYSTEM - HOURLY+ TIMEFRAMES
# =============================================================================

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data")
}

# CORE PARAMETERS
INITIAL_CAPITAL = 100000
POSITION_SIZE = 0.02
YEAR_FILTER = 2025

# ML TIMEFRAME CONFIGURATIONS
ML_TIMEFRAME_CONFIGS = {
    '1hour': {
        'source_data': '1hour',
        'profit_target': 0.035,     # 3.5%
        'stop_loss': 0.020,         # 2.0%
        'max_hold_bars': 24,        # 24 hours
        'lookback_periods': 20,     # Features lookback
        'description': 'ML 1-hour trading'
    },
    '4hour': {
        'source_data': '1hour',     # Aggregate from 1-hour
        'aggregation': 4,
        'profit_target': 0.050,     # 5.0%
        'stop_loss': 0.025,         # 2.5%
        'max_hold_bars': 12,        # 48 hours
        'lookback_periods': 15,     # Features lookback
        'description': 'ML 4-hour trading'
    },
    'daily': {
        'source_data': 'daily',
        'profit_target': 0.060,     # 6.0%
        'stop_loss': 0.030,         # 3.0%
        'max_hold_bars': 10,        # 10 days
        'lookback_periods': 20,     # Features lookback
        'description': 'ML daily trading'
    },
    'weekly': {
        'source_data': 'daily',
        'resample_freq': 'W-MON',
        'profit_target': 0.080,     # 8.0%
        'stop_loss': 0.040,         # 4.0%
        'max_hold_bars': 8,         # 8 weeks
        'lookback_periods': 12,     # Features lookback
        'description': 'ML weekly trading'
    }
}

# ML MODELS TO TEST
ML_MODELS = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42, probability=True)
}

def get_all_symbols():
    """Get all available symbols"""
    try:
        files = [f for f in os.listdir(DATA_PATHS['daily']) if f.endswith('.csv')]
        return sorted([f.replace('.csv', '') for f in files])
    except:
        return []

def load_and_prepare_data(symbol, timeframe):
    """Load and prepare data for ML analysis"""
    try:
        config = ML_TIMEFRAME_CONFIGS[timeframe]

        if timeframe == '4hour':
            # Load 1-hour data and aggregate to 4-hour
            file_path = os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv")
            if not os.path.exists(file_path):
                return None

            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)

            if len(df) < 100:
                return None

            # Aggregate to 4-hour bars
            df.set_index('date', inplace=True)
            df_resampled = df.groupby(df.index // 4).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index(drop=True)

        elif timeframe == 'weekly':
            # Load daily data and resample to weekly
            file_path = os.path.join(DATA_PATHS['daily'], f"{symbol}.csv")
            if not os.path.exists(file_path):
                return None

            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)

            df.set_index('date', inplace=True)
            df_resampled = df.resample('W-MON').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna().reset_index()

        else:  # 1hour or daily
            data_source = config['source_data']
            file_path = os.path.join(DATA_PATHS[data_source], f"{symbol}.csv")
            if not os.path.exists(file_path):
                return None

            df_resampled = pd.read_csv(file_path)
            df_resampled['date'] = pd.to_datetime(df_resampled['date'])
            df_resampled = df_resampled[df_resampled['date'].dt.year == YEAR_FILTER].reset_index(drop=True)

        if len(df_resampled) < 50:
            return None

        return df_resampled

    except Exception as e:
        print(f"Error loading {symbol} for {timeframe}: {e}")
        return None

def create_comprehensive_features(df, lookback_periods=20):
    """Create comprehensive ML features"""
    try:
        features_df = df.copy()

        # Price-based features
        features_df['returns'] = df['close'].pct_change()
        features_df['high_low_ratio'] = df['high'] / df['low']
        features_df['close_open_ratio'] = df['close'] / df['open']

        # Technical indicators
        features_df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        features_df['macd'] = ta.trend.macd_diff(df['close'])
        features_df['bb_upper'], features_df['bb_middle'], features_df['bb_lower'] = ta.volatility.bollinger_hband(df['close']), ta.volatility.bollinger_mavg(df['close']), ta.volatility.bollinger_lband(df['close'])
        features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / features_df['bb_middle']
        features_df['bb_position'] = (df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])

        # Trend indicators
        features_df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        features_df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        features_df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])

        # Moving averages
        for period in [5, 10, 20]:
            features_df[f'sma_{period}'] = df['close'].rolling(period).mean()
            features_df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
            features_df[f'price_vs_sma_{period}'] = df['close'] / features_df[f'sma_{period}']

        # Volume indicators
        features_df['volume_sma'] = df['volume'].rolling(20).mean()
        features_df['volume_ratio'] = df['volume'] / features_df['volume_sma']
        features_df['volume_price_trend'] = ta.volume.volume_price_trend(df['close'], df['volume'])
        features_df['acc_dist_index'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])

        # Volatility indicators
        features_df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        features_df['volatility'] = df['returns'].rolling(10).std()

        # Momentum indicators
        for period in [5, 10, 15]:
            features_df[f'momentum_{period}'] = df['close'] / df['close'].shift(period)
            features_df[f'roc_{period}'] = ta.momentum.roc(df['close'], window=period)

        # Lag features (past values)
        for lag in range(1, min(lookback_periods//4, 6)):
            features_df[f'close_lag_{lag}'] = df['close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)

        # Statistical features (rolling)
        features_df['price_mean_reversion'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        features_df['volume_mean_reversion'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

        # Market structure features
        features_df['higher_highs'] = (df['high'] > df['high'].shift(1)).astype(int).rolling(5).sum()
        features_df['lower_lows'] = (df['low'] < df['low'].shift(1)).astype(int).rolling(5).sum()

        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')

        return features_df

    except Exception as e:
        print(f"Error creating features: {e}")
        return df

def create_target_variable(df, timeframe):
    """Create target variable for ML (future returns)"""
    try:
        config = ML_TIMEFRAME_CONFIGS[timeframe]
        profit_target = config['profit_target']
        stop_loss = config['stop_loss']
        max_hold = config['max_hold_bars']

        targets = []

        for i in range(len(df) - max_hold):
            entry_price = df.iloc[i]['close']

            # Look forward to find exit
            for j in range(1, max_hold + 1):
                if i + j >= len(df):
                    break

                future_price = df.iloc[i + j]['close']
                return_pct = (future_price - entry_price) / entry_price

                if return_pct >= profit_target:
                    targets.append(1)  # Profitable trade
                    break
                elif return_pct <= -stop_loss:
                    targets.append(0)  # Loss trade
                    break
            else:
                # Max hold reached
                final_price = df.iloc[i + max_hold]['close'] if i + max_hold < len(df) else df.iloc[-1]['close']
                final_return = (final_price - entry_price) / entry_price
                targets.append(1 if final_return > 0 else 0)

        # Pad to match dataframe length
        targets.extend([0] * max_hold)

        return np.array(targets)

    except Exception as e:
        print(f"Error creating target variable: {e}")
        return np.zeros(len(df))

def train_ml_models(features_df, targets, feature_columns):
    """Train multiple ML models and select best"""
    try:
        # Prepare data
        X = features_df[feature_columns].values
        y = targets

        # Remove any remaining NaN or inf values
        mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
        X = X[mask]
        y = y[mask]

        if len(X) < 100:  # Need minimum samples
            return None, None, None

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

        best_model = None
        best_score = 0
        best_model_name = None

        model_results = {}

        # Train and evaluate each model
        for model_name, model in ML_MODELS.items():
            try:
                # Train model
                model.fit(X_train, y_train)

                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                avg_cv_score = cv_scores.mean()

                # Test score
                test_score = model.score(X_test, y_test)

                model_results[model_name] = {
                    'cv_score': avg_cv_score,
                    'test_score': test_score,
                    'model': model
                }

                if avg_cv_score > best_score:
                    best_score = avg_cv_score
                    best_model = model
                    best_model_name = model_name

            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue

        return best_model, scaler, best_model_name, model_results

    except Exception as e:
        print(f"Error in ML training: {e}")
        return None, None, None, {}

def backtest_ml_system(df, model, scaler, feature_columns, timeframe):
    """Backtest ML system"""
    try:
        config = ML_TIMEFRAME_CONFIGS[timeframe]

        cash = INITIAL_CAPITAL
        positions = {}
        trades = []

        # Create features for entire dataset
        features_df = create_comprehensive_features(df, config['lookback_periods'])

        # Start after sufficient lookback
        start_idx = config['lookback_periods'] + 10

        for i in range(start_idx, len(df) - config['max_hold_bars']):
            current = df.iloc[i]

            # Exit existing positions
            positions_to_close = []
            for pos_id, pos in positions.items():
                current_return = (current['close'] - pos['entry_price']) / pos['entry_price']
                bars_held = i - pos['entry_bar']

                # Exit conditions
                exit_reason = None
                if current_return >= config['profit_target']:
                    exit_reason = 'Profit Target'
                elif current_return <= -config['stop_loss']:
                    exit_reason = 'Stop Loss'
                elif bars_held >= config['max_hold_bars']:
                    exit_reason = 'Time Exit'

                if exit_reason:
                    shares = pos['shares']
                    exit_price = current['close'] * 0.999

                    pnl = (exit_price - pos['entry_price']) * shares
                    commission = (pos['entry_price'] + exit_price) * shares * 0.0003
                    net_pnl = pnl - commission

                    trade = {
                        'symbol': pos['symbol'],
                        'timeframe': timeframe,
                        'entry_date': pos['entry_date'],
                        'exit_date': current['date'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': net_pnl,
                        'return_pct': current_return,
                        'exit_reason': exit_reason,
                        'bars_held': bars_held,
                        'ml_prediction': pos['ml_prediction']
                    }

                    trades.append(trade)
                    cash += pos['entry_value'] + net_pnl
                    positions_to_close.append(pos_id)

            for pos_id in positions_to_close:
                del positions[pos_id]

            # Check for new entries using ML prediction
            if len(positions) == 0 and cash > 10000:
                try:
                    # Prepare features for current bar
                    current_features = features_df.iloc[i][feature_columns].values.reshape(1, -1)

                    # Check for NaN or inf values
                    if not (np.isnan(current_features).any() or np.isinf(current_features).any()):
                        # Scale features
                        current_features_scaled = scaler.transform(current_features)

                        # Get ML prediction
                        prediction = model.predict(current_features_scaled)[0]
                        prediction_proba = model.predict_proba(current_features_scaled)[0][1]  # Probability of success

                        # Only trade if ML predicts success with high confidence
                        if prediction == 1 and prediction_proba > 0.6:
                            position_value = cash * POSITION_SIZE
                            entry_price = current['close'] * 1.001
                            shares = position_value / entry_price

                            if cash >= position_value:
                                positions[0] = {
                                    'symbol': 'ML_TRADE',
                                    'entry_date': current['date'],
                                    'entry_price': entry_price,
                                    'shares': shares,
                                    'entry_bar': i,
                                    'entry_value': position_value,
                                    'ml_prediction': prediction_proba
                                }
                                cash -= position_value

                except Exception as e:
                    continue  # Skip this bar if features can't be calculated

        return trades

    except Exception as e:
        print(f"Error in ML backtesting: {e}")
        return []

def run_comprehensive_ml_analysis():
    """Run comprehensive ML analysis across multiple timeframes"""

    print("🤖 COMPREHENSIVE ML TRADING SYSTEM ANALYSIS")
    print("=" * 60)
    print("Testing ML models on 1-hour, 4-hour, daily, and weekly timeframes")
    print("This will take some time due to ML model training...")
    print()

    # Get symbols (test subset for speed)
    all_symbols = get_all_symbols()
    test_symbols = all_symbols[:50]  # Test 50 symbols

    all_results = {}

    # Test each timeframe
    for timeframe in ['1hour', '4hour', 'daily', 'weekly']:
        config = ML_TIMEFRAME_CONFIGS[timeframe]

        print(f"\n🤖 TESTING ML {timeframe.upper()} TIMEFRAME")
        print(f"Target: {config['profit_target']*100:.1f}% profit, {config['stop_loss']*100:.1f}% stop")
        print(f"Description: {config['description']}")
        print("-" * 70)

        timeframe_results = []
        successful = 0
        failed = 0

        for i, symbol in enumerate(test_symbols, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(test_symbols)} symbols...")

            try:
                # Load data
                df = load_and_prepare_data(symbol, timeframe)
                if df is None:
                    failed += 1
                    continue

                # Create features
                features_df = create_comprehensive_features(df, config['lookback_periods'])

                # Create target variable
                targets = create_target_variable(df, timeframe)

                # Get feature columns (exclude date and OHLCV)
                feature_columns = [col for col in features_df.columns 
                                 if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]

                if len(feature_columns) < 10:  # Need minimum features
                    failed += 1
                    continue

                # Train ML models
                best_model, scaler, best_model_name, model_results = train_ml_models(
                    features_df, targets, feature_columns
                )

                if best_model is None:
                    failed += 1
                    continue

                # Backtest with best model
                trades = backtest_ml_system(df, best_model, scaler, feature_columns, timeframe)

                if len(trades) >= 2:
                    df_trades = pd.DataFrame(trades)
                    winning_trades = len(df_trades[df_trades['pnl'] > 0])

                    result = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'best_model': best_model_name,
                        'total_trades': len(trades),
                        'winning_trades': winning_trades,
                        'win_rate': (winning_trades / len(trades)) * 100,
                        'total_pnl': df_trades['pnl'].sum(),
                        'avg_pnl_per_trade': df_trades['pnl'].mean(),
                        'best_trade': df_trades['pnl'].max(),
                        'worst_trade': df_trades['pnl'].min(),
                        'avg_ml_confidence': df_trades['ml_prediction'].mean(),
                        'bars_analyzed': len(df),
                        'is_profitable': df_trades['pnl'].sum() > 0,
                        'model_scores': model_results
                    }

                    timeframe_results.append(result)
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                failed += 1
                continue

        # Store results
        all_results[timeframe] = timeframe_results

        # Summarize timeframe
        if timeframe_results:
            df_results = pd.DataFrame(timeframe_results)
            profitable_count = len(df_results[df_results['is_profitable']])
            total_pnl = df_results['total_pnl'].sum()
            total_trades = df_results['total_trades'].sum()
            avg_win_rate = df_results['win_rate'].mean()
            roc = (total_pnl / INITIAL_CAPITAL) * 100

            print(f"\n✅ ML {timeframe.upper()} RESULTS:")
            print(f"  Symbols tested: {successful}")
            print(f"  Profitable symbols: {profitable_count} ({profitable_count/successful*100:.1f}%)")
            print(f"  Total trades: {total_trades}")
            print(f"  Total PnL: ₹{total_pnl:,.0f}")
            print(f"  Return on Capital: {roc:.2f}%")
            print(f"  Average win rate: {avg_win_rate:.1f}%")

            # Show breakthrough if profitable
            if roc > 0:
                print(f"\n🎉 ML BREAKTHROUGH! {timeframe.upper()} IS PROFITABLE!")
                print(f"   Positive ROC: {roc:.2f}%")
            elif roc > -2:
                print(f"\n🎯 VERY CLOSE! ML {timeframe.upper()} nearly profitable")
                print(f"   ROC: {roc:.2f}%")

            # Save results
            df_results.to_csv(f'ml_{timeframe}_results.csv', index=False)

            # Show top ML performers
            if len(df_results) > 0:
                top_performers = df_results.nlargest(5, 'total_pnl')
                print(f"\n🏆 TOP 5 ML {timeframe.upper()} PERFORMERS:")
                for _, row in top_performers.iterrows():
                    print(f"  {row['symbol']:<12}: ₹{row['total_pnl']:>6.0f} "
                          f"({row['total_trades']:>2} trades, {row['win_rate']:>5.1f}% win, "
                          f"{row['best_model']})")
        else:
            print(f"\n❌ ML {timeframe.upper()}: No successful results")

    # FINAL ML ANALYSIS
    print(f"\n\n🤖 COMPREHENSIVE ML ANALYSIS RESULTS")
    print("=" * 60)

    if any(all_results.values()):
        print(f"\n📊 ML PERFORMANCE BY TIMEFRAME:")
        print("-" * 35)
        print(f"{'Timeframe':<12} {'Symbols':<8} {'Success%':<9} {'Total PnL':<12} {'ROC%':<8} {'Status'}")
        print("-" * 70)

        ml_summary = []

        for timeframe, results in all_results.items():
            if results:
                df_results = pd.DataFrame(results)
                profitable_count = len(df_results[df_results['is_profitable']])
                success_rate = (profitable_count / len(df_results)) * 100
                total_pnl = df_results['total_pnl'].sum()
                roc = (total_pnl / INITIAL_CAPITAL) * 100

                status = "🎉 PROFIT!" if roc > 0 else "🎯 Close" if roc > -2 else "❌ Loss"

                ml_summary.append((timeframe, len(df_results), success_rate, total_pnl, roc, status))

                print(f"{timeframe:<12} {len(df_results):<8} {success_rate:<8.1f}% "
                      f"₹{total_pnl:<11.0f} {roc:<7.1f}% {status}")

        # Find best ML timeframe
        if ml_summary:
            best_ml = max(ml_summary, key=lambda x: x[4])  # Best ROC

            print(f"\n🏆 BEST ML TIMEFRAME: {best_ml[0].upper()}")
            print(f"   ROC: {best_ml[4]:.2f}%")
            print(f"   Success Rate: {best_ml[2]:.1f}%")
            print(f"   Total PnL: ₹{best_ml[3]:.0f}")

            if best_ml[4] > 0:
                print(f"\n🚀 ML BREAKTHROUGH ACHIEVED!")
                print(f"Machine Learning shows profitability on {best_ml[0]} timeframe!")
                print(f"Recommended: Deploy ML {best_ml[0]} system for live trading")
            elif best_ml[4] > -5:
                print(f"\n🎯 ML SHOWS PROMISE!")
                print(f"Close to profitability - consider parameter optimization")
            else:
                print(f"\n📊 ML RESULTS:")
                print(f"All ML timeframes show losses, but methodology is comprehensive")

    else:
        print("\n❌ No ML results generated across any timeframe")

    return all_results

if __name__ == "__main__":
    results = run_comprehensive_ml_analysis()
