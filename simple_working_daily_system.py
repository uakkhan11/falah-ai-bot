
import os
import pandas as pd
import numpy as np
import talib
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# SIMPLE WORKING DAILY TRADING SYSTEM
# =============================================================================

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data")
}

# Simple, working parameters
INITIAL_CAPITAL = 100000
MAX_POSITIONS = 3
BASE_RISK = 0.02  # 2% risk per trade
ML_THRESHOLD = 0.6  # Reasonable threshold
YEAR_FILTER = 2025

# Simple features that work
FEATURES = ["rsi", "adx", "volume_ratio", "atr_ratio"]

# =============================================================================
# SIMPLE DATA LOADING
# =============================================================================

def load_simple_data(symbol):
    """Load data with simple error handling"""
    try:
        m15 = pd.read_csv(os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"))
        m15['date'] = pd.to_datetime(m15['date'])
        m15 = m15[m15['date'].dt.year == YEAR_FILTER].reset_index(drop=True)

        if len(m15) < 1000:
            return None

        return m15
    except:
        return None

# =============================================================================
# SIMPLE INDICATORS
# =============================================================================

def add_simple_indicators(df):
    """Add simple, reliable indicators"""
    try:
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        # ADX
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

        # Volume ratio
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # ATR ratio
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_ratio'] = df['atr'] / df['close']

        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df
    except Exception as e:
        print(f"Error adding indicators: {e}")
        return df

# =============================================================================
# SIMPLE SIGNAL GENERATION
# =============================================================================

def generate_simple_signals(df):
    """Generate simple, reliable signals"""
    try:
        df['signal'] = 0

        for i in range(50, len(df)):
            current = df.iloc[i]

            # Simple breakout signal
            high_20 = df['high'].iloc[i-20:i].max()
            volume_avg = df['volume'].iloc[i-20:i].mean()

            breakout_signal = (
                current['close'] > high_20 and           # Price breakout
                current['volume'] > volume_avg * 1.5 and # Volume confirmation
                current['rsi'] < 80 and                  # Not overbought
                current['adx'] > 20                      # Some trend strength
            )

            if breakout_signal:
                df.iloc[i, df.columns.get_loc('signal')] = 1

        return df
    except Exception as e:
        print(f"Error generating signals: {e}")
        df['signal'] = 0
        return df

# =============================================================================
# SIMPLE ML MODEL
# =============================================================================

def train_simple_ml(df):
    """Train simple, working ML model"""
    try:
        # Create simple labels - shorter lookforward to get more labels
        df['future_return'] = df['close'].shift(-2) / df['close'] - 1
        df['label'] = (df['future_return'] > 0.01).astype(int)  # 1% threshold

        # Remove future data
        df = df[:-2].copy()

        # Check if we have the features
        missing_features = [f for f in FEATURES if f not in df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            return None, {}

        # Prepare data
        df_clean = df[FEATURES + ['label']].dropna()

        if len(df_clean) < 500:
            print(f"Insufficient clean data: {len(df_clean)}")
            return None, {}

        X = df_clean[FEATURES]
        y = df_clean['label']

        # Check label balance
        pos_rate = y.mean()
        print(f"Positive label rate: {pos_rate:.1%}")

        if pos_rate < 0.10 or pos_rate > 0.90:
            print(f"Imbalanced labels: {pos_rate:.1%}")
            return None, {}

        # Simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False, random_state=42
        )

        # Simple XGBoost model (compatible with all versions)
        model = XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=50,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Simple validation
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred, zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'samples': len(X_train),
            'pos_rate': pos_rate
        }

        print(f"Model trained: {accuracy:.3f} accuracy, {precision:.3f} precision")

        return model, metrics

    except Exception as e:
        print(f"Error training ML model: {e}")
        return None, {}

# =============================================================================
# SIMPLE BACKTESTING
# =============================================================================

def simple_backtest(df, symbol, model):
    """Simple, working backtesting"""
    try:
        cash = INITIAL_CAPITAL
        positions = {}
        trades = []

        # Get ML predictions
        if model is not None:
            feature_data = df[FEATURES].fillna(method='ffill').fillna(0)
            ml_proba = model.predict_proba(feature_data)[:, 1]
            df['ml_proba'] = ml_proba
        else:
            df['ml_proba'] = 0.5

        for i in range(1, len(df)):
            current = df.iloc[i]

            # Close existing positions
            positions_to_close = []
            for pos_id, pos in positions.items():
                current_return = (current['close'] - pos['entry_price']) / pos['entry_price']
                bars_held = i - pos['entry_bar']

                # Simple exit rules
                exit_reason = None
                if current_return >= 0.03:  # 3% profit
                    exit_reason = 'Take Profit'
                elif current_return <= -0.02:  # 2% loss
                    exit_reason = 'Stop Loss'
                elif bars_held >= 20:  # Time exit
                    exit_reason = 'Time Exit'

                if exit_reason:
                    # Calculate trade
                    shares = pos['shares']
                    exit_price = current['close'] * 0.9995  # Simple slippage

                    pnl = (exit_price - pos['entry_price']) * shares
                    commission = (pos['entry_price'] + exit_price) * shares * 0.0005
                    net_pnl = pnl - commission

                    trade = {
                        'symbol': symbol,
                        'entry_date': pos['entry_date'],
                        'exit_date': current['date'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': shares,
                        'pnl': net_pnl,
                        'return_pct': current_return,
                        'exit_reason': exit_reason,
                        'bars_held': bars_held
                    }

                    trades.append(trade)
                    cash += pos['entry_value'] + net_pnl
                    positions_to_close.append(pos_id)

            # Remove closed positions
            for pos_id in positions_to_close:
                del positions[pos_id]

            # Check for new entries
            if (current['signal'] == 1 and
                current['ml_proba'] >= ML_THRESHOLD and
                len(positions) < MAX_POSITIONS and
                cash > 5000):

                # Simple position sizing
                position_value = cash * BASE_RISK
                entry_price = current['close'] * 1.0005  # Simple slippage
                shares = position_value / entry_price

                if cash >= position_value:
                    positions[len(positions)] = {
                        'entry_date': current['date'],
                        'entry_price': entry_price,
                        'shares': shares,
                        'entry_bar': i,
                        'entry_value': position_value
                    }
                    cash -= position_value

        return trades

    except Exception as e:
        print(f"Error in backtest: {e}")
        return []

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_simple_daily_backtest():
    """Run simple, working daily backtest"""

    print("💡 SIMPLE WORKING DAILY TRADING SYSTEM")
    print("=" * 50)
    print("Goal: Get something that actually works!")
    print()

    # Get symbols
    try:
        symbols = [f.replace('.csv', '') for f in os.listdir(DATA_PATHS['daily']) 
                  if f.endswith('.csv')][:15]
        print(f"Testing {len(symbols)} symbols")
    except:
        print("Error loading symbols")
        return

    all_results = []
    successful_symbols = 0

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

        try:
            # Load data
            df = load_simple_data(symbol)
            if df is None:
                print(f"  ✗ No data for {symbol}")
                continue

            # Add indicators
            df = add_simple_indicators(df)

            # Generate signals
            df = generate_simple_signals(df)

            # Check if we have any signals
            signal_count = df['signal'].sum()
            if signal_count < 10:
                print(f"  ✗ Only {signal_count} signals generated")
                continue

            # Train ML model on first 70% of data
            split_point = int(len(df) * 0.7)
            train_data = df.iloc[:split_point].copy()
            test_data = df.iloc[split_point:].copy()

            model, ml_metrics = train_simple_ml(train_data)

            if model is None:
                print(f"  ✗ ML model training failed")
                continue

            # Run backtest on last 30%
            trades = simple_backtest(test_data, symbol, model)

            if len(trades) >= 5:  # Need minimum trades
                # Calculate stats
                df_trades = pd.DataFrame(trades)
                winning_trades = len(df_trades[df_trades['pnl'] > 0])

                result = {
                    'symbol': symbol,
                    'total_trades': len(trades),
                    'winning_trades': winning_trades,
                    'win_rate': winning_trades / len(trades) * 100,
                    'total_pnl': df_trades['pnl'].sum(),
                    'avg_pnl_per_trade': df_trades['pnl'].mean(),
                    'best_trade': df_trades['pnl'].max(),
                    'worst_trade': df_trades['pnl'].min(),
                    'ml_accuracy': ml_metrics.get('accuracy', 0),
                    'ml_precision': ml_metrics.get('precision', 0)
                }

                all_results.append(result)
                successful_symbols += 1

                print(f"  ✅ {result['total_trades']} trades, "
                      f"{result['win_rate']:.1f}% win rate, "
                      f"₹{result['total_pnl']:.0f} PnL")
            else:
                print(f"  ✗ Only {len(trades)} trades generated")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    # Display results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('total_pnl', ascending=False)

        print(f"\n\n🎯 SIMPLE SYSTEM RESULTS:")
        print("=" * 40)
        print(f"Successfully processed: {successful_symbols}/{len(symbols)} symbols")
        print()

        print("TOP PERFORMERS:")
        print("-" * 30)
        display_cols = ['symbol', 'total_trades', 'win_rate', 'total_pnl', 'ml_accuracy']
        print(results_df[display_cols].head(10).round(2).to_string(index=False))

        # Overall stats
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Trades: {results_df['total_trades'].sum()}")
        print(f"  Average Win Rate: {results_df['win_rate'].mean():.1f}%")
        print(f"  Total PnL: ₹{results_df['total_pnl'].sum():.0f}")
        print(f"  Average ML Accuracy: {results_df['ml_accuracy'].mean():.1%}")
        print(f"  Profitable Symbols: {len(results_df[results_df['total_pnl'] > 0])}")

        # Save results
        results_df.to_csv('simple_working_results.csv', index=False)
        print(f"\nResults saved to: simple_working_results.csv")

        return results_df
    else:
        print("\nNo successful results. Check your data paths and files.")
        return None

if __name__ == "__main__":
    results = run_simple_daily_backtest()
