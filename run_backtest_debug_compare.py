import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================
# Configurations
# ======================
CSV_PATH = "your_training_data.csv"  # Update with your data path
MODEL_PATH = "model.pkl"  # Path to your trained model
TARGET_COLUMN = "outcome"

# Trading parameters
INITIAL_CAPITAL = 100000
POSITION_SIZE = 0.1  # Use 10% of capital per trade
STOP_LOSS_PCT = 0.05  # 5% stop loss
TAKE_PROFIT_PCT = 0.10  # 10% take profit

# ======================
# Load Data and Model
# ======================
print("Loading data and model...")
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower() for c in df.columns]

# Load the trained model
model = joblib.load(MODEL_PATH)

# Parse dates
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.tz_localize(None)  # Remove timezone
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

print(f"Loaded {len(df)} rows of data")

# ======================
# Calculate Features (matching model training)
# ======================
print("Calculating technical indicators...")

# Only calculate indicators that can be computed from available data
if 'rsi' not in df.columns:
    df['rsi'] = ta.rsi(df['close'], length=14)

if 'ema10' not in df.columns:
    df['ema10'] = ta.ema(df['close'], length=10)

if 'ema21' not in df.columns:
    df['ema21'] = ta.ema(df['close'], length=21)

# Calculate MACD for additional signals (even though not in final features)
macd_data = ta.macd(df['close'])
df['macd_hist'] = macd_data['MACDh_12_26_9']
df['macd_signal'] = macd_data['MACDs_12_26_9']

# Define features that match the trained model
features = ['rsi', 'atr', 'adx', 'ema10', 'ema21', 'volumechange']
available_features = [f for f in features if f in df.columns]

print(f"Available features for prediction: {available_features}")

# Drop initial rows with NaN values from indicators
df = df.dropna(subset=available_features).reset_index(drop=True)

# ======================
# Generate Trading Signals
# ======================
print("Generating trading signals...")

# Prepare features for model prediction
X = df[available_features]

# Get model predictions
df['ml_signal'] = model.predict(X)
df['ml_probability'] = model.predict_proba(X)[:, 1]  # Probability of positive class

# Create additional rule-based signals for comparison
df['rsi_signal'] = 0
df.loc[(df['rsi'] < 30) & (df['rsi'].shift(1) >= 30), 'rsi_signal'] = 1  # RSI oversold buy
df.loc[(df['rsi'] > 70) & (df['rsi'].shift(1) <= 70), 'rsi_signal'] = -1  # RSI overbought sell

# Combined strategy: ML + RSI confirmation
df['combined_signal'] = 0
df.loc[(df['ml_signal'] == 1) & (df['rsi'] < 60), 'combined_signal'] = 1  # ML buy with RSI not overbought
df.loc[(df['ml_signal'] == 0) & (df['rsi'] > 70), 'combined_signal'] = -1  # ML no-buy with RSI overbought

# ======================
# Backtesting Engine
# ======================
def run_backtest(df, signal_column, initial_capital=INITIAL_CAPITAL):
    """
    Simple backtesting engine for close-price-only data
    """
    results = []
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_date = None
    
    for i in range(1, len(df)):
        current_date = df.loc[i, 'date'] if 'date' in df.columns else i
        current_price = df.loc[i, 'close']
        signal = df.loc[i, signal_column]
        
        # Exit logic
        if position != 0:
            # Calculate returns
            if position > 0:  # Long position
                pct_change = (current_price - entry_price) / entry_price
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                if pct_change <= -STOP_LOSS_PCT:
                    should_exit = True
                    exit_reason = "Stop Loss"
                elif pct_change >= TAKE_PROFIT_PCT:
                    should_exit = True
                    exit_reason = "Take Profit"
                elif signal == -1:
                    should_exit = True
                    exit_reason = "Signal Exit"
                    
                if should_exit:
                    # Calculate trade result
                    trade_value = position * current_price
                    profit_loss = trade_value - (position * entry_price)
                    capital += profit_loss
                    
                    results.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position_size': position,
                        'profit_loss': profit_loss,
                        'return_pct': pct_change,
                        'exit_reason': exit_reason,
                        'capital': capital
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_date = None
        
        # Entry logic
        if position == 0 and signal == 1:
            # Calculate position size
            position_value = capital * POSITION_SIZE
            position = position_value / current_price
            entry_price = current_price
            entry_date = current_date
    
    return pd.DataFrame(results)

# ======================
# Run Backtests
# ======================
print("Running backtests...")

# Backtest ML strategy
ml_results = run_backtest(df, 'ml_signal')

# Backtest RSI strategy  
rsi_results = run_backtest(df, 'rsi_signal')

# Backtest combined strategy
combined_results = run_backtest(df, 'combined_signal')

# ======================
# Performance Metrics
# ======================
def calculate_metrics(results_df, strategy_name):
    """Calculate performance metrics for a backtest"""
    if len(results_df) == 0:
        return f"No trades for {strategy_name} strategy"
    
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['profit_loss'] > 0])
    losing_trades = len(results_df[results_df['profit_loss'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    total_profit = results_df['profit_loss'].sum()
    avg_return = results_df['return_pct'].mean()
    max_return = results_df['return_pct'].max()
    min_return = results_df['return_pct'].min()
    
    final_capital = INITIAL_CAPITAL + total_profit
    total_return = total_profit / INITIAL_CAPITAL
    
    return f"""
{strategy_name} Strategy Results:
================================
Total Trades: {total_trades}
Winning Trades: {winning_trades}
Losing Trades: {losing_trades}  
Win Rate: {win_rate:.2%}
Total P&L: ${total_profit:,.2f}
Total Return: {total_return:.2%}
Average Return per Trade: {avg_return:.2%}
Best Trade: {max_return:.2%}
Worst Trade: {min_return:.2%}
Final Capital: ${final_capital:,.2f}
"""

# Print results
print(calculate_metrics(ml_results, "ML"))
print(calculate_metrics(rsi_results, "RSI"))  
print(calculate_metrics(combined_results, "Combined"))

# ======================
# Save Results
# ======================
if len(ml_results) > 0:
    ml_results.to_csv('ml_backtest_results.csv', index=False)
    
if len(rsi_results) > 0:
    rsi_results.to_csv('rsi_backtest_results.csv', index=False)
    
if len(combined_results) > 0:
    combined_results.to_csv('combined_backtest_results.csv', index=False)

print("\nBacktesting complete! Results saved to CSV files.")
print("\nNote: This backtest uses close-price-only data with simplified assumptions.")
print("Consider adding more realistic trading costs, slippage, and market hours restrictions for production use.")
