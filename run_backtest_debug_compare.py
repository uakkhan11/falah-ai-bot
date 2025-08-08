import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================
# Optimized Balanced Configurations
# ======================
CSV_PATH = "your_training_data.csv"
MODEL_PATH = "model.pkl"

# Trading parameters (in Indian Rupees)
INITIAL_CAPITAL = 1000000  # ₹10 Lakhs starting capital
FIXED_POSITION_SIZE = 100000  # ₹1 Lakh per trade
INITIAL_STOP_LOSS_PCT = 0.05  # 5% initial stop loss
TRAILING_STOP_PCT = 0.035  # 3.5% trailing stop (balanced)
TAKE_PROFIT_PCT = 0.10  # 10% take profit target
PROFIT_TRAIL_TRIGGER = 0.04  # Start trailing after 4% profit (earlier trigger)
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

# Data cleaning (same as before)
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
# Balanced Feature Calculation
# ======================
features = ['rsi', 'atr', 'adx', 'ema10', 'ema21', 'volumechange']
available_features = [f for f in features if f in df.columns]

# Add ONLY essential trend indicators (not overloading)
if 'sma50' not in df.columns:
    df['sma50'] = ta.sma(df['close'], length=50)

# Calculate MACD for trend confirmation (simplified)
macd_data = ta.macd(df['close'])
df['macd_hist'] = macd_data['MACDh_12_26_9']

print(f"Using features: {available_features}")
df = df.dropna(subset=available_features + ['sma50']).reset_index(drop=True)

# ======================
# Optimized Signal Generation
# ======================
print("Generating optimized balanced signals...")
X = df[available_features]

df['ml_signal'] = model.predict(X)
df['ml_probability'] = model.predict_proba(X)[:, 1]

# Balanced trend identification (simpler but effective)
df['trend_bullish'] = df['close'] > df['sma50']  # Simple but effective
df['price_momentum'] = df['close'] > df['close'].shift(3)  # Shorter momentum period

# ======================
# OPTIMIZED RSI STRATEGY (Balanced Approach)
# ======================
df['rsi_signal'] = 0

# RSI Buy: Balanced conditions - not too restrictive
rsi_buy_condition = (
    (df['rsi'] < 40) &  # Slightly less oversold for more opportunities
    (df['rsi'] > df['rsi'].shift(1)) &  # RSI turning upward
    df['trend_bullish']  # Only major trend filter (removed complex filters)
)
df.loc[rsi_buy_condition, 'rsi_signal'] = 1

# RSI Sell: Earlier exits for better risk management
rsi_sell_condition = (
    (df['rsi'] > 65) |  # Earlier overbought exit
    (~df['trend_bullish'])  # Exit when trend turns bearish
)
df.loc[rsi_sell_condition, 'rsi_signal'] = -1

# ======================
# OPTIMIZED COMBINED STRATEGY
# ======================
df['combined_signal'] = 0

# Combined Buy: ML confidence + basic RSI filter
combined_buy_condition = (
    (df['ml_signal'] == 1) & 
    (df['ml_probability'] > 0.60) &  # Reasonable confidence threshold
    (df['rsi'] < 70) &  # Not overbought
    df['trend_bullish']  # Basic trend confirmation
)
df.loc[combined_buy_condition, 'combined_signal'] = 1

# Combined Sell: Conservative exit conditions
combined_sell_condition = (
    (df['rsi'] > 70) |  # Overbought
    (~df['trend_bullish']) |  # Trend reversal
    (df['ml_probability'] < 0.4)  # ML confidence drops
)
df.loc[combined_sell_condition, 'combined_signal'] = -1

# ======================
# Enhanced Trailing Stop Backtesting Engine
# ======================
def optimized_trailing_backtest(df, signal_column, initial_capital=INITIAL_CAPITAL):
    """
    Optimized backtesting with balanced trailing stops
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
        
        # Enhanced exit logic with optimized trailing stops
        if position_shares > 0 and entry_price > 0:
            # Update highest price
            if current_price > highest_price_since_entry:
                highest_price_since_entry = current_price
            
            pct_change = (current_price - entry_price) / entry_price
            
            # Activate trailing stops earlier (at 4% instead of 5%)
            if not profit_trail_active and pct_change >= PROFIT_TRAIL_TRIGGER:
                profit_trail_active = True
                trailing_stop_price = highest_price_since_entry * (1 - TRAILING_STOP_PCT)
                print(f"Trailing activated: Entry ₹{entry_price:.2f}, Current ₹{current_price:.2f}, Profit: {pct_change*100:.2f}%")
            
            # Update trailing stop
            if profit_trail_active:
                new_trailing_stop = highest_price_since_entry * (1 - TRAILING_STOP_PCT)
                if new_trailing_stop > trailing_stop_price:
                    trailing_stop_price = new_trailing_stop
            
            should_exit = False
            exit_reason = ""
            
            # Optimized exit conditions
            if current_price <= initial_stop_loss:
                should_exit = True
                exit_reason = "Initial Stop Loss"
            elif profit_trail_active and current_price <= trailing_stop_price:
                should_exit = True
                exit_reason = "Trailing Stop"
                print(f"Trailing stop hit: ₹{current_price:.2f} <= ₹{trailing_stop_price:.2f}")
            elif pct_change >= TAKE_PROFIT_PCT:
                should_exit = True
                exit_reason = "Take Profit Target"
            elif signal == -1:
                should_exit = True
                exit_reason = "Signal Exit"
            
            if should_exit:
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
                    'trailing_trigger_price': trailing_stop_price if profit_trail_active else 0,
                    'portfolio_value': cash
                })
                
                # Reset variables
                position_shares = 0
                entry_price = 0
                entry_date = None
                highest_price_since_entry = 0
                initial_stop_loss = 0
                trailing_stop_price = 0
                profit_trail_active = False
                trade_count += 1
        
        # Entry logic (unchanged)
        elif position_shares == 0 and signal == 1 and cash >= FIXED_POSITION_SIZE:
            position_cost = FIXED_POSITION_SIZE * (1 + TRANSACTION_COST)
            
            if cash >= position_cost:
                position_shares = FIXED_POSITION_SIZE / current_price
                entry_price = current_price
                entry_date = current_date
                highest_price_since_entry = current_price
                initial_stop_loss = entry_price * (1 - INITIAL_STOP_LOSS_PCT)
                cash -= position_cost
    
    # Close remaining positions
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
            'trailing_trigger_price': trailing_stop_price if profit_trail_active else 0,
            'portfolio_value': cash
        })
    
    return pd.DataFrame(results), cash

# ======================
# Run Optimized Backtests
# ======================
print("Running optimized balanced backtests...")

ml_results, ml_final_cash = optimized_trailing_backtest(df, 'ml_signal')
rsi_results, rsi_final_cash = optimized_trailing_backtest(df, 'rsi_signal')
combined_results, combined_final_cash = optimized_trailing_backtest(df, 'combined_signal')

# ======================
# Enhanced Performance Metrics with Comparison
# ======================
def calculate_optimized_metrics(results_df, final_cash, strategy_name):
    """Calculate comprehensive metrics with trailing stop effectiveness"""
    if len(results_df) == 0:
        return f"\n{strategy_name} Strategy: No trades executed"
    
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['profit_loss'] > 0])
    win_rate = winning_trades / total_trades
    
    # Detailed exit analysis
    exit_reasons = results_df['exit_reason'].value_counts()
    trailing_stops = len(results_df[results_df['exit_reason'] == 'Trailing Stop'])
    profit_targets = len(results_df[results_df['exit_reason'] == 'Take Profit Target'])
    signal_exits = exit_reasons.get('Signal Exit', 0)
    stop_losses = exit_reasons.get('Initial Stop Loss', 0)
    
    total_profit = results_df['profit_loss'].sum()
    avg_return = results_df['return_pct'].mean()
    max_return = results_df['return_pct'].max()
    min_return = results_df['return_pct'].min()
    
    total_return = (final_cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Risk metrics
    std_return = results_df['return_pct'].std()
    sharpe = avg_return / std_return if std_return > 0 else 0
    
    # Max drawdown calculation
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
    trailing_profit_trades = trailing_trades[trailing_trades['profit_loss'] > 0]
    trailing_effectiveness = len(trailing_profit_trades) / len(trailing_trades) if len(trailing_trades) > 0 else 0
    
    return f"""
{strategy_name} Strategy Results (Optimized):
===========================================
📊 TRADE SUMMARY:
Total Trades: {total_trades}
Winning Trades: {winning_trades}
Win Rate: {win_rate:.2%}

💰 FINANCIAL PERFORMANCE:
Total P&L: ₹{total_profit:,.2f}
Total Return: {total_return:.2%}
Avg Return/Trade: {avg_return:.2f}%
Best Trade: {max_return:.2f}%
Worst Trade: {min_return:.2f}%
Final Portfolio: ₹{final_cash:,.2f}

⚠️ RISK METRICS:
Volatility: {std_return:.2f}%
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_drawdown:.2f}%

🎯 EXIT ANALYSIS:
• Trailing Stops: {trailing_stops} ({trailing_stops/total_trades*100:.1f}%)
• Profit Targets: {profit_targets} ({profit_targets/total_trades*100:.1f}%)
• Signal Exits: {signal_exits} ({signal_exits/total_trades*100:.1f}%)
• Stop Losses: {stop_losses} ({stop_losses/total_trades*100:.1f}%)

🚀 TRAILING STOP EFFECTIVENESS:
• Trades with Trailing: {len(trailing_trades)}
• Trailing Success Rate: {trailing_effectiveness:.1%}
• Avg Trailing Return: {trailing_trades['return_pct'].mean():.2f}%
"""

# Print optimized results
print("\n" + "="*70)
print("OPTIMIZED BALANCED BACKTEST RESULTS")
print("="*70)

print(calculate_optimized_metrics(ml_results, ml_final_cash, "ML"))
print(calculate_optimized_metrics(rsi_results, rsi_final_cash, "Optimized RSI"))
print(calculate_optimized_metrics(combined_results, combined_final_cash, "Optimized Combined"))

# ======================
# Performance Comparison Summary
# ======================
strategies_summary = {
    'ML': {'trades': len(ml_results), 'win_rate': len(ml_results[ml_results['profit_loss'] > 0])/len(ml_results) if len(ml_results) > 0 else 0, 'final': ml_final_cash},
    'RSI': {'trades': len(rsi_results), 'win_rate': len(rsi_results[rsi_results['profit_loss'] > 0])/len(rsi_results) if len(rsi_results) > 0 else 0, 'final': rsi_final_cash},
    'Combined': {'trades': len(combined_results), 'win_rate': len(combined_results[combined_results['profit_loss'] > 0])/len(combined_results) if len(combined_results) > 0 else 0, 'final': combined_final_cash}
}

print(f"\n🏆 STRATEGY RANKING:")
ranked_strategies = sorted(strategies_summary.items(), key=lambda x: x[1]['final'], reverse=True)
for i, (strategy, stats) in enumerate(ranked_strategies, 1):
    return_pct = (stats['final'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"{i}. {strategy}: ₹{stats['final']:,.0f} ({return_pct:.1f}% return, {stats['win_rate']:.1%} win rate, {stats['trades']} trades)")

# ======================
# Save Optimized Results
# ======================
if len(ml_results) > 0:
    ml_results.to_csv('ml_optimized_backtest.csv', index=False)

if len(rsi_results) > 0:
    rsi_results.to_csv('rsi_optimized_backtest.csv', index=False)
    
if len(combined_results) > 0:
    combined_results.to_csv('combined_optimized_backtest.csv', index=False)

print(f"\n✅ OPTIMIZATION COMPLETE!")
print("Key Optimizations Applied:")
print("• Balanced RSI conditions (40/65 thresholds)")
print("• Simplified trend filters (SMA50 only)")
print("• Earlier trailing activation (4% vs 5%)")
print("• Moderate ML confidence (60% vs 70%)")
print("• Enhanced exit tracking and analysis")
print("• Results saved with '_optimized' suffix")
