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

# Trading parameters (in Indian Rupees)
INITIAL_CAPITAL = 1000000  # ₹10 Lakhs starting capital
FIXED_POSITION_SIZE = 100000  # ₹1 Lakh per trade
STOP_LOSS_PCT = 0.05  # 5% stop loss
TAKE_PROFIT_PCT = 0.10  # 10% take profit
MIN_PRICE = 50.0  # Minimum ₹50 per share
MAX_PRICE = 10000.0  # Maximum ₹10,000 per share
TRANSACTION_COST = 0.001  # 0.1% transaction cost per trade (brokerage + taxes)

# ======================
# Load and Clean Data
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

# Strict data cleaning
print("Cleaning data...")
initial_rows = len(df)

# Remove extreme prices and ensure clean data
df = df[
    (df['close'] >= MIN_PRICE) & 
    (df['close'] <= MAX_PRICE) & 
    (df['close'].notna()) &
    (np.isfinite(df['close']))
].copy()

# Remove obvious data errors (huge price jumps)
df['price_change'] = df['close'].pct_change().abs()
df = df[df['price_change'] < 0.5].copy()  # Remove >50% single-day moves

print(f"Cleaned dataset: {len(df)} rows (removed {initial_rows - len(df)} rows)")
print(f"Price range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")

# ======================
# Calculate Features
# ======================
features = ['rsi', 'atr', 'adx', 'ema10', 'ema21', 'volumechange']
available_features = [f for f in features if f in df.columns]

print(f"Using features: {available_features}")
df = df.dropna(subset=available_features).reset_index(drop=True)

# ======================
# Generate Signals
# ======================
print("Generating signals...")
X = df[available_features]

df['ml_signal'] = model.predict(X)
df['ml_probability'] = model.predict_proba(X)[:, 1]

# Conservative RSI signals
df['rsi_signal'] = 0
df.loc[(df['rsi'] < 25) & (df['rsi'].shift(1) >= 25), 'rsi_signal'] = 1  # Very oversold
df.loc[(df['rsi'] > 75) & (df['rsi'].shift(1) <= 75), 'rsi_signal'] = -1  # Very overbought

# High-confidence ML signals only
df['combined_signal'] = 0
df.loc[(df['ml_signal'] == 1) & (df['ml_probability'] > 0.7), 'combined_signal'] = 1
df.loc[(df['ml_signal'] == 0) & (df['ml_probability'] < 0.3), 'combined_signal'] = -1

# ======================
# Realistic Backtesting Engine
# ======================
def realistic_backtest(df, signal_column, initial_capital=INITIAL_CAPITAL):
    """
    Realistic backtesting with fixed position sizes and transaction costs (in INR)
    """
    results = []
    cash = initial_capital
    position_shares = 0
    entry_price = 0
    entry_date = None
    trade_count = 0
    max_trades = 200  # Reasonable limit
    
    for i in range(1, len(df)):
        if trade_count >= max_trades:
            break
            
        current_date = df.loc[i, 'date'] if 'date' in df.columns else i
        current_price = df.loc[i, 'close']
        signal = df.loc[i, signal_column]
        
        # Skip invalid data
        if current_price <= 0 or not np.isfinite(current_price):
            continue
        
        # Exit logic
        if position_shares > 0 and entry_price > 0:
            current_value = position_shares * current_price
            pct_change = (current_price - entry_price) / entry_price
            
            should_exit = False
            exit_reason = ""
            
            # Exit conditions
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
                # Calculate realistic exit
                exit_value = current_value * (1 - TRANSACTION_COST)  # Transaction cost
                profit_loss = exit_value - FIXED_POSITION_SIZE
                cash += exit_value
                
                results.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': position_shares,
                    'profit_loss': profit_loss,
                    'return_pct': pct_change * 100,
                    'exit_reason': exit_reason,
                    'portfolio_value': cash
                })
                
                position_shares = 0
                entry_price = 0
                entry_date = None
                trade_count += 1
        
        # Entry logic
        elif position_shares == 0 and signal == 1 and cash >= FIXED_POSITION_SIZE:
            # Use fixed position size
            position_cost = FIXED_POSITION_SIZE * (1 + TRANSACTION_COST)
            
            if cash >= position_cost:
                position_shares = FIXED_POSITION_SIZE / current_price
                entry_price = current_price
                entry_date = current_date
                cash -= position_cost
    
    # Close any remaining position
    if position_shares > 0:
        final_value = position_shares * df.iloc[-1]['close'] * (1 - TRANSACTION_COST)
        profit_loss = final_value - FIXED_POSITION_SIZE
        cash += final_value
        
        results.append({
            'entry_date': entry_date,
            'exit_date': df.iloc[-1]['date'] if 'date' in df.columns else len(df)-1,
            'entry_price': entry_price,
            'exit_price': df.iloc[-1]['close'],
            'shares': position_shares,
            'profit_loss': profit_loss,
            'return_pct': (df.iloc[-1]['close'] - entry_price) / entry_price * 100,
            'exit_reason': "End of Period",
            'portfolio_value': cash
        })
    
    return pd.DataFrame(results), cash

# ======================
# Run Realistic Backtests
# ======================
print("Running realistic backtests...")

ml_results, ml_final_cash = realistic_backtest(df, 'ml_signal')
rsi_results, rsi_final_cash = realistic_backtest(df, 'rsi_signal')  
combined_results, combined_final_cash = realistic_backtest(df, 'combined_signal')

# ======================
# Realistic Performance Metrics
# ======================
def calculate_realistic_metrics(results_df, final_cash, strategy_name):
    """Calculate realistic performance metrics in Indian Rupees"""
    if len(results_df) == 0:
        return f"\n{strategy_name} Strategy: No trades executed"
    
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['profit_loss'] > 0])
    win_rate = winning_trades / total_trades
    
    total_profit = results_df['profit_loss'].sum()
    avg_return = results_df['return_pct'].mean()
    max_return = results_df['return_pct'].max()
    min_return = results_df['return_pct'].min()
    
    total_return = (final_cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Risk metrics
    std_return = results_df['return_pct'].std()
    sharpe = avg_return / std_return if std_return >
