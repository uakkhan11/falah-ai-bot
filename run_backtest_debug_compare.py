import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================
# Enhanced Configurations
# ======================
CSV_PATH = "your_training_data.csv"
MODEL_PATH = "model.pkl"

# Trading parameters (in Indian Rupees)
INITIAL_CAPITAL = 1000000  # ₹10 Lakhs starting capital
FIXED_POSITION_SIZE = 100000  # ₹1 Lakh per trade
INITIAL_STOP_LOSS_PCT = 0.05  # 5% initial stop loss
TRAILING_STOP_PCT = 0.03  # 3% trailing stop (tighter than initial)
TAKE_PROFIT_PCT = 0.10  # 10% take profit target
PROFIT_TRAIL_TRIGGER = 0.05  # Start trailing after 5% profit
MIN_PRICE = 50.0  # Minimum ₹50 per share
MAX_PRICE = 10000.0  # Maximum ₹10,000 per share
TRANSACTION_COST = 0.001  # 0.1% transaction cost per trade

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

df = df[
    (df['close'] >= MIN_PRICE) & 
    (df['close'] <= MAX_PRICE) & 
    (df['close'].notna()) &
    (np.isfinite(df['close']))
].copy()

df['price_change'] = df['close'].pct_change().abs()
df = df[df['price_change'] < 0.5].copy()

print(f"Cleaned dataset: {len(df)} rows (removed {initial_rows - len(df)} rows)")
print(f"Price range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")

# ======================
# Calculate Enhanced Features
# ======================
features = ['rsi', 'atr', 'adx', 'ema10', 'ema21', 'volumechange']
available_features = [f for f in features if f in df.columns]

# Add trend indicators for better RSI signals
if 'sma50' not in df.columns:
    df['sma50'] = ta.sma(df['close'], length=50)
if 'sma200' not in df.columns:
    df['sma200'] = ta.sma(df['close'], length=200)

# Calculate MACD for trend confirmation
macd_data = ta.macd(df['close'])
df['macd_line'] = macd_data['MACD_12_26_9']
df['macd_signal_line'] = macd_data['MACDs_12_26_9']
df['macd_hist'] = macd_data['MACDh_12_26_9']

print(f"Using features: {available_features}")
df = df.dropna(subset=available_features + ['sma50', 'sma200']).reset_index(drop=True)

# ======================
# Enhanced Signal Generation
# ======================
print("Generating enhanced signals...")
X = df[available_features]

df['ml_signal'] = model.predict(X)
df['ml_probability'] = model.predict_proba(X)[:, 1]

# Enhanced RSI signals with trend filtering
df['trend_bullish'] = (df['close'] > df['sma50']) & (df['sma50'] > df['sma200'])
df['trend_bearish'] = (df['close'] < df['sma50']) & (df['sma50'] < df['sma200'])
df['price_momentum'] = df['close'] > df['close'].shift(5)  # 5-period momentum

# Improved RSI strategy - only buy in uptrends with momentum
df['rsi_signal'] = 0

# RSI Buy: Oversold + Bullish trend + Positive momentum + RSI turning up
rsi_buy_condition = (
    (df['rsi'] < 35) &  # Slightly less oversold for more signals
    (df['rsi'] > df['rsi'].shift(1)) &  # RSI turning upward
    (df['rsi'].shift(1) < df['rsi'].shift(2)) &  # RSI was declining before
    df['trend_bullish'] &  # Only in bullish trends
    df['price_momentum'] &  # Positive price momentum
    (df['macd_hist'] > df['macd_hist'].shift(1))  # MACD histogram improving
)
df.loc[rsi_buy_condition, 'rsi_signal'] = 1

# RSI Sell: Overbought OR trend reversal
rsi_sell_condition = (
    (df['rsi'] > 70) | 
    (df['trend_bearish']) |
    (df['macd_hist'] < df['macd_hist'].shift(1))
)
df.loc[rsi_sell_condition, 'rsi_signal'] = -1

# Enhanced Combined strategy
df['combined_signal'] = 0
combined_buy_condition = (
    (df['ml_signal'] == 1) & 
    (df['ml_probability'] > 0.65) &  # High confidence ML signal
    (df['rsi'] < 65) &  # Not overbought
    df['trend_bullish'] &  # Bullish trend
    (df['macd_line'] > df['macd_signal_line'])  # MACD bullish
)
df.loc[combined_buy_condition, 'combined_signal'] = 1

combined_sell_condition = (
    (df['rsi'] > 75) | 
    (df['trend_bearish']) |
    (df['macd_line'] < df['macd_signal_line'])
)
df.loc[combined_sell_condition, 'combined_signal'] = -1

# ======================
# Enhanced Backtesting Engine with Trailing Stops
# ======================
def enhanced_trailing_backtest(df, signal_column, initial_capital=INITIAL_CAPITAL):
    """
    Advanced backtesting with trailing stops and profit booking
    """
    results = []
    cash = initial_capital
    position_shares = 0
    entry_price = 0
    entry_date = None
    highest_price_since_entry = 0
    initial_stop_loss = 0
    trailing_stop_price = 0
    profit_trail_active = False
    trade_count = 0
    max_trades = 200
    
    for i in range(1, len(df)):
        if trade_count >= max_trades:
            break
            
        current_date = df.loc[i, 'date'] if 'date' in df.columns else i
        current_price = df.loc[i, 'close']
        signal = df.loc[i, signal_column]
        
        if current_price <= 0 or not np.isfinite(current_price):
            continue
        
        # Exit logic with trailing stops
        if position_shares > 0 and entry_price > 0:
            # Update highest price since entry
            if current_price > highest_price_since_entry:
                highest_price_since_entry = current_price
            
            pct_change = (current_price - entry_price) / entry_price
            
            # Activate profit trailing after reaching trigger threshold
            if not profit_trail_active and pct_change >= PROFIT_TRAIL_TRIGGER:
                profit_trail_active = True
                trailing_stop_price = highest_price_since_entry * (1 - TRAILING_STOP_PCT)
            
            # Update trailing stop if profit trailing is active
            if profit_trail_active:
                new_trailing_stop = highest_price_since_entry * (1 - TRAILING_STOP_PCT)
                trailing_stop_price = max(trailing_stop_price, new_trailing_stop)
            
            should_exit = False
            exit_reason = ""
            
            # Exit conditions
            if current_price <= initial_stop_loss:
                should_exit = True
                exit_reason = "Initial Stop Loss"
            elif profit_trail_active and current_price <= trailing_stop_price:
                should_exit = True
                exit_reason = "Trailing Stop"
            elif pct_change >= TAKE_PROFIT_PCT:
                should_exit = True
                exit_reason = "Take Profit Target"
            elif signal == -1:
                should_exit = True
                exit_reason = "Signal Exit"
            
            if should_exit:
                # Calculate realistic exit
                exit_value = position_shares * current_price * (1 - TRANSACTION_COST)
                profit_loss = exit_value - FIXED_POSITION_SIZE
                cash += exit_value
                
                results.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'highest_price': highest_price_since_entry,
                    'shares': position_shares,
                    'profit_loss': profit_loss,
                    'return_pct': pct_change * 100,
                    'exit_reason': exit_reason,
                    'trailing_active': profit_trail_active,
                    'portfolio_value': cash
                })
                
                # Reset position variables
                position_shares = 0
                entry_price = 0
                entry_date = None
                highest_price_since_entry = 0
                initial_stop_loss = 0
                trailing_stop_price = 0
                profit_trail_active = False
                trade_count += 1
        
        # Entry logic
        elif position_shares == 0 and signal == 1 and cash >= FIXED_POSITION_SIZE:
            position_cost = FIXED_POSITION_SIZE * (1 + TRANSACTION_COST)
            
            if cash >= position_cost:
                position_shares = FIXED_POSITION_SIZE / current_price
                entry_price = current_price
                entry_date = current_date
                highest_price_since_entry = current_price
                initial_stop_loss = entry_price * (1 - INITIAL_STOP_LOSS_PCT)
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
            'highest_price': highest_price_since_entry,
            'shares': position_shares,
            'profit_loss': profit_loss,
            'return_pct': (df.iloc[-1]['close'] - entry_price) / entry_price * 100,
            'exit_reason': "End of Period",
            'trailing_active': profit_trail_active,
            'portfolio_value': cash
        })
    
    return pd.DataFrame(results), cash

# ======================
# Run Enhanced Backtests
# ======================
print("Running enhanced backtests with trailing stops...")

ml_results, ml_final_cash = enhanced_trailing_backtest(df, 'ml_signal')
rsi_results, rsi_final_cash = enhanced_trailing_backtest(df, 'rsi_signal')
combined_results, combined_final_cash = enhanced_trailing_backtest(df, 'combined_signal')

# ======================
# Enhanced Performance Metrics
# ======================
def calculate_enhanced_metrics(results_df, final_cash, strategy_name):
    """Calculate enhanced performance metrics with trailing stop analysis"""
    if len(results_df) == 0:
        return f"\n{strategy_name} Strategy: No trades executed"
    
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['profit_loss'] > 0])
    win_rate = winning_trades / total_trades
    
    # Exit reason analysis
    exit_reasons = results_df['exit_reason'].value_counts()
    trailing_stops = len(results_df[results_df['exit_reason'] == 'Trailing Stop'])
    profit_targets = len(results_df[results_df['exit_reason'] == 'Take Profit Target'])
    
    total_profit = results_df['profit_loss'].sum()
    avg_return = results_df['return_pct'].mean()
    max_return = results_df['return_pct'].max()
    min_return = results_df['return_pct'].min()
    
    total_return = (final_cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Enhanced risk metrics
    std_return = results_df['return_pct'].std()
    if std_return > 0:
        sharpe = avg_return / std_return
    else:
        sharpe = 0
    
    # Calculate max drawdown
    portfolio_values = results_df['portfolio_value'].values
    max_drawdown = 0
    if len(portfolio_values) > 0:
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        max_drawdown = max_drawdown * 100
    
    # Trailing stop effectiveness
    trailing_trades = results_df[results_df['trailing_active'] == True]
    avg_trailing_profit = trailing_trades['return_pct'].mean() if len(trailing_trades) > 0 else 0
    
    return f"""
{strategy_name} Strategy Results (Enhanced):
==========================================
Total Trades: {total_trades}
Winning Trades: {winning_trades}
Win Rate: {win_rate:.2%}
Total P&L: ₹{total_profit:,.2f}
Total Return: {total_return:.2%}
Avg Return/Trade: {avg_return:.2f}%
Best Trade: {max_return:.2f}%
Worst Trade: {min_return:.2f}%
Volatility: {std_return:.2f}%
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_drawdown:.2f}%
Final Portfolio: ₹{final_cash:,.2f}

Exit Reason Analysis:
- Trailing Stops: {trailing_stops} ({trailing_stops/total_trades*100:.1f}%)
- Profit Targets: {profit_targets} ({profit_targets/total_trades*100:.1f}%)
- Signal Exits: {exit_reasons.get('Signal Exit', 0)}
- Initial Stop Loss: {exit_reasons.get('Initial Stop Loss', 0)}

Trailing Stop Performance:
- Trades with Trailing: {len(trailing_trades)}
- Avg Trailing Profit: {avg_trailing_profit:.2f}%
"""

# Print enhanced results
print(calculate_enhanced_metrics(ml_results, ml_final_cash, "ML"))
print(calculate_enhanced_metrics(rsi_results, rsi_final_cash, "Enhanced RSI"))
print(calculate_enhanced_metrics(combined_results, combined_final_cash, "Enhanced Combined"))

# ======================
# Save Enhanced Results
# ======================
if len(ml_results) > 0:
    ml_results.to_csv('ml_enhanced_trailing_backtest.csv', index=False)

if len(rsi_results) > 0:
    rsi_results.to_csv('rsi_enhanced_trailing_backtest.csv', index=False)
    
if len(combined_results) > 0:
    combined_results.to_csv('combined_enhanced_trailing_backtest.csv', index=False)

print("\n" + "="*60)
print("ENHANCED TRAILING STOP BACKTESTING COMPLETE")
print("="*60)
print("Key Enhancements:")
print("• Trailing stop loss: 3% (tighter than 5% initial)")
print("• Profit trailing: Activated after 5% gain")
print("• Take profit target: 10% maintained")
print("• Enhanced RSI: Trend-filtered with momentum")
print("• Better signals: MACD + SMA trend confirmation")
print("• Detailed exit analysis included")
print("• All results saved with '_enhanced_trailing' suffix")
