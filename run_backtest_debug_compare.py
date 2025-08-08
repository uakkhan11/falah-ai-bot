import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================
# Configurations
# ======================
CSV_PATH = "your_training_data.csv"
MODEL_PATH = "model.pkl"

# Trading parameters
INITIAL_CAPITAL = 100000
POSITION_SIZE = 0.05  # Reduced to 5% of capital per trade for safety
STOP_LOSS_PCT = 0.05  # 5% stop loss
TAKE_PROFIT_PCT = 0.10  # 10% take profit
MIN_PRICE = 10.0  # Minimum price filter to avoid penny stock issues
MAX_POSITION_SIZE = 0.2  # Maximum 20% of capital per trade

# ======================
# Load Data and Model
# ======================
print("Loading data and model...")
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower() for c in df.columns]

model = joblib.load(MODEL_PATH)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.tz_localize(None)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

print(f"Loaded {len(df)} rows of data")

# ======================
# Data Quality Checks
# ======================
print("Performing data quality checks...")

# Remove rows with invalid prices
initial_rows = len(df)
df = df[df['close'] > MIN_PRICE].copy()
df = df[df['close'] < 100000].copy()  # Remove extremely high prices
df = df[df['close'].notna()].copy()

print(f"Removed {initial_rows - len(df)} rows with invalid prices")
print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# ======================
# Calculate Features
# ======================
print("Calculating technical indicators...")

if 'rsi' not in df.columns:
    df['rsi'] = ta.rsi(df['close'], length=14)

if 'ema10' not in df.columns:
    df['ema10'] = ta.ema(df['close'], length=10)

if 'ema21' not in df.columns:
    df['ema21'] = ta.ema(df['close'], length=21)

features = ['rsi', 'atr', 'adx', 'ema10', 'ema21', 'volumechange']
available_features = [f for f in features if f in df.columns]

print(f"Available features: {available_features}")

# Clean data
df = df.dropna(subset=available_features + ['close']).reset_index(drop=True)

# ======================
# Generate Signals
# ======================
print("Generating trading signals...")

X = df[available_features]
df['ml_signal'] = model.predict(X)
df['ml_probability'] = model.predict_proba(X)[:, 1]

# RSI signals with additional filters
df['rsi_signal'] = 0
df.loc[(df['rsi'] < 30) & (df['rsi'].shift(1) >= 30) & (df['close'] > df['ema21']), 'rsi_signal'] = 1
df.loc[(df['rsi'] > 70) & (df['rsi'].shift(1) <= 70), 'rsi_signal'] = -1

# Combined strategy with probability threshold
df['combined_signal'] = 0
df.loc[(df['ml_signal'] == 1) & (df['ml_probability'] > 0.6) & (df['rsi'] < 65), 'combined_signal'] = 1
df.loc[(df['rsi'] > 75), 'combined_signal'] = -1

# ======================
# Improved Backtesting Engine
# ======================
def run_backtest_safe(df, signal_column, initial_capital=INITIAL_CAPITAL):
    """
    Safe backtesting engine with better risk management
    """
    results = []
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_date = None
    max_trades = 1000  # Limit number of trades
    
    for i in range(1, min(len(df), len(df))):
        if len(results) >= max_trades:
            break
            
        current_date = df.loc[i, 'date'] if 'date' in df.columns else i
        current_price = df.loc[i, 'close']
        signal = df.loc[i, signal_column]
        
        # Skip if price is invalid
        if current_price <= 0 or np.isinf(current_price) or np.isnan(current_price):
            continue
        
        # Exit logic
        if position != 0 and entry_price > 0:
            pct_change = (current_price - entry_price) / entry_price
            
            # Clamp extreme returns
            pct_change = max(min(pct_change, 5.0), -0.99)  # Cap at 500% gain, -99% loss
            
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
                # Calculate safe trade result
                position_value = min(position * entry_price, capital * MAX_POSITION_SIZE)
                profit_loss = position_value * pct_change
                
                # Safety check
                if abs(profit_loss) < capital * 2:  # Don't allow P&L > 200% of capital
                    capital += profit_loss
                    
                    results.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position_size': position,
                        'profit_loss': profit_loss,
                        'return_pct': pct_change * 100,  # Convert to percentage
                        'exit_reason': exit_reason,
                        'capital': capital
                    })
                
                position = 0
                entry_price = 0
                entry_date = None
        
        # Entry logic with safety checks
        if position == 0 and signal == 1 and capital > 0:
            position_value = capital * POSITION_SIZE
            if position_value > MIN_PRICE and current_price > MIN_PRICE:
                position = position_value / current_price
                entry_price = current_price
                entry_date = current_date
    
    return pd.DataFrame(results)

# ======================
# Run Safe Backtests
# ======================
print("Running safe backtests...")

ml_results = run_backtest_safe(df, 'ml_signal')
rsi_results = run_backtest_safe(df, 'rsi_signal')
combined_results = run_backtest_safe(df, 'combined_signal')

# ======================
# Safe Performance Metrics
# ======================
def calculate_safe_metrics(results_df, strategy_name):
    """Calculate safe performance metrics"""
    if len(results_df) == 0:
        return f"No trades executed for {strategy_name} strategy"
    
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['profit_loss'] > 0])
    losing_trades = len(results_df[results_df['profit_loss'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    total_profit = results_df['profit_loss'].sum()
    avg_return = results_df['return_pct'].mean()
    max_return = results_df['return_pct'].max()
    min_return = results_df['return_pct'].min()
    
    final_capital = INITIAL_CAPITAL + total_profit
    total_return = total_profit / INITIAL_CAPITAL * 100
    
    # Additional risk metrics
    std_return = results_df['return_pct'].std()
    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
    
    return f"""
{strategy_name} Strategy Results:
================================
Total Trades: {total_trades}
Winning Trades: {winning_trades}
Losing Trades: {losing_trades}
Win Rate: {win_rate:.2%}
Total P&L: ${total_profit:,.2f}
Total Return: {total_return:.2%}
Average Return per Trade: {avg_return:.2f}%
Best Trade: {max_return:.2f}%
Worst Trade: {min_return:.2f}%
Return Std Dev: {std_return:.2f}%
Sharpe Ratio: {sharpe_ratio:.2f}
Final Capital: ${final_capital:,.2f}
"""

# Print results
print(calculate_safe_metrics(ml_results, "ML"))
print(calculate_safe_metrics(rsi_results, "RSI"))
print(calculate_safe_metrics(combined_results, "Combined"))

# Save results
if len(ml_results) > 0:
    ml_results.to_csv('ml_backtest_safe.csv', index=False)
    
if len(rsi_results) > 0:
    rsi_results.to_csv('rsi_backtest_safe.csv', index=False)
    
if len(combined_results) > 0:
    combined_results.to_csv('combined_backtest_safe.csv', index=False)

print("\nSafe backtesting complete!")
print("Key improvements:")
print("- Data quality filters applied")  
print("- Position sizing limits enforced")
print("- Return capping to prevent infinite values")
print("- Additional risk metrics included")
