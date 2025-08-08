import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================
# Enhanced 15% Target Configurations
# ======================
CSV_PATH = "your_training_data.csv"
MODEL_PATH = "model.pkl"

# Enhanced Trading parameters (in Indian Rupees)
INITIAL_CAPITAL = 1000000  # ₹10 Lakhs starting capital
FIXED_POSITION_SIZE = 100000  # ₹1 Lakh per trade
INITIAL_STOP_LOSS_PCT = 0.05  # 5% initial stop loss
TAKE_PROFIT_PCT = 0.15  # 15% take profit target (INCREASED)
PROFIT_TRAIL_TRIGGER = 0.08  # Start trailing after 8% profit (LOWERED)
ATR_MULTIPLIER = 2.0  # Tighter ATR trailing (REDUCED from 2.5)
MIN_PRICE = 50.0
MAX_PRICE = 10000.0
TRANSACTION_COST = 0.001

# ======================
# Load and Clean Data
# ======================
print("Loading data and model for 15% target enhanced strategies...")
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

# ======================
# Enhanced Technical Indicators (Top 3 Only)
# ======================
features = ['rsi', 'atr', 'adx', 'ema10', 'ema21', 'volumechange']
available_features = [f for f in features if f in df.columns]

# Calculate ATR for dynamic trailing stops
if 'atr' not in df.columns:
    df['atr'] = df['close'].rolling(14).apply(lambda x: x.std() * 1.5)

# WILLIAMS %R STRATEGY (Best RSI Alternative)
df['williams_r'] = ((df['close'].rolling(14).max() - df['close']) / 
                   (df['close'].rolling(14).max() - df['close'].rolling(14).min())) * -100

# SUPERTREND STRATEGY (Trend Following)
df['hl2'] = (df['close'] + df['close'].shift(1)) / 2
supertrend_data = ta.supertrend(df['close'], df['close'], df['close'], length=10, multiplier=3)
if supertrend_data is not None:
    df['supertrend'] = supertrend_data['SUPERT_10_3.0']
    df['supertrend_direction'] = supertrend_data['SUPERTd_10_3.0']
else:
    # Fallback SuperTrend calculation
    df['supertrend'] = df['close']
    df['supertrend_direction'] = np.where(df['close'] > df['close'].shift(10), 1, -1)

print(f"Enhanced features available: {available_features}")
df = df.dropna(subset=available_features + ['atr', 'williams_r']).reset_index(drop=True)

# ======================
# Enhanced Strategy Signal Generation (Top 3 Only)
# ======================
print("Generating enhanced signals for TOP 3 strategies only...")
X = df[available_features]

df['ml_signal'] = model.predict(X)
df['ml_probability'] = model.predict_proba(X)[:, 1]

# ENHANCED ML STRATEGY (with higher confidence)
df['enhanced_ml_signal'] = 0
enhanced_ml_buy_condition = (
    (df['ml_signal'] == 1) & 
    (df['ml_probability'] > 0.70)  # Higher confidence for 15% targets
)
df.loc[enhanced_ml_buy_condition, 'enhanced_ml_signal'] = 1

enhanced_ml_sell_condition = (
    (df['ml_signal'] == 0) & 
    (df['ml_probability'] < 0.35)
)
df.loc[enhanced_ml_sell_condition, 'enhanced_ml_signal'] = -1

# ENHANCED WILLIAMS %R STRATEGY (Best RSI Replacement)
df['enhanced_williams_signal'] = 0
enhanced_williams_buy_condition = (
    (df['williams_r'] < -80) &  # Oversold
    (df['williams_r'] > df['williams_r'].shift(1)) &  # Starting to turn up
    (df['williams_r'].shift(1) < df['williams_r'].shift(2)) &  # Was declining
    (df['close'] > df['close'].shift(3))  # Short-term momentum
)
df.loc[enhanced_williams_buy_condition, 'enhanced_williams_signal'] = 1

enhanced_williams_sell_condition = (
    (df['williams_r'] > -15) |  # Overbought exit
    (df['close'] < df['close'].shift(5))  # Momentum fading
)
df.loc[enhanced_williams_sell_condition, 'enhanced_williams_signal'] = -1

# ENHANCED SUPERTREND STRATEGY
df['enhanced_supertrend_signal'] = 0
if 'supertrend_direction' in df.columns:
    enhanced_supertrend_buy_condition = (
        (df['supertrend_direction'] == 1) &  # SuperTrend bullish
        (df['close'] > df['close'].shift(2))  # Price momentum
    )
    df.loc[enhanced_supertrend_buy_condition, 'enhanced_supertrend_signal'] = 1
    
    enhanced_supertrend_sell_condition = (df['supertrend_direction'] == -1)
    df.loc[enhanced_supertrend_sell_condition, 'enhanced_supertrend_signal'] = -1

# COMBINED TOP PERFORMERS STRATEGY
df['top_performers_combined_signal'] = 0
top_combined_buy_condition = (
    (df['enhanced_ml_signal'] == 1) &  # ML confidence
    (
        (df['enhanced_williams_signal'] == 1) |  # Williams %R confirmation
        (df['enhanced_supertrend_signal'] == 1)   # OR SuperTrend confirmation
    )
)
df.loc[top_combined_buy_condition, 'top_performers_combined_signal'] = 1

top_combined_sell_condition = (
    (df['enhanced_williams_signal'] == -1) |
    (df['enhanced_supertrend_signal'] == -1)
)
df.loc[top_combined_sell_condition, 'top_performers_combined_signal'] = -1

# ======================
# Advanced ATR-Based Trailing Stop Engine (Enhanced)
# ======================
def enhanced_15pct_atr_backtest(df, signal_column, initial_capital=INITIAL_CAPITAL):
    """
    Enhanced backtesting with 15% targets and improved ATR trailing stops
    """
    results = []
    cash = initial_capital
    position_shares = 0
    entry_price = 0
    entry_date = None
    highest_price_since_entry = 0
    initial_stop_loss = 0
    atr_trailing_stop = 0
    profit_trail_active = False
    trade_count = 0
    max_trades = 200
    trailing_activations = 0
    trailing_exits = 0
    
    for i in range(1, len(df)):
        if trade_count >= max_trades:
            break
            
        current_date = df.loc[i, 'date'] if 'date' in df.columns else i
        current_price = df.loc[i, 'close']
        current_atr = df.loc[i, 'atr']
        signal = df.loc[i, signal_column]
        
        if current_price <= 0 or not np.isfinite(current_price) or pd.isna(current_atr):
            continue
        
        # Enhanced exit logic with improved ATR trailing
        if position_shares > 0 and entry_price > 0:
            if current_price > highest_price_since_entry:
                highest_price_since_entry = current_price
            
            pct_change = (current_price - entry_price) / entry_price
            
            # Enhanced ATR trailing activation (8% instead of 6%)
            if not profit_trail_active and pct_change >= PROFIT_TRAIL_TRIGGER:
                profit_trail_active = True
                atr_trailing_stop = current_price - (current_atr * ATR_MULTIPLIER)
                trailing_activations += 1
                print(f"ATR trailing activated: Entry Rs{entry_price:.2f}, Current Rs{current_price:.2f}, Profit: {pct_change*100:.2f}%")
            
            # Update ATR trailing stop (tighter 2.0x multiplier)
            if profit_trail_active:
                new_atr_stop = current_price - (current_atr * ATR_MULTIPLIER)
                if new_atr_stop > atr_trailing_stop:
                    atr_trailing_stop = new_atr_stop
            
            should_exit = False
            exit_reason = ""
            
            # Enhanced exit conditions
            if current_price <= initial_stop_loss:
                should_exit = True
                exit_reason = "Initial Stop Loss"
            elif profit_trail_active and current_price <= atr_trailing_stop:
                should_exit = True
                exit_reason = "ATR Trailing Stop"
                trailing_exits += 1
                print(f"ATR trailing exit: Rs{current_price:.2f} <= Rs{atr_trailing_stop:.2f}, Profit secured: {pct_change*100:.2f}%")
            elif pct_change >= TAKE_PROFIT_PCT:
                should_exit = True
                exit_reason = "Take Profit Target (15%)"
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
                    'atr_trail_active': profit_trail_active,
                    'atr_trail_price': atr_trailing_stop if profit_trail_active else 0,
                    'max_profit_pct': (highest_price_since_entry - entry_price) / entry_price * 100,
                    'portfolio_value': cash
                })
                
                # Reset variables
                position_shares = 0
                entry_price = 0
                entry_date = None
                highest_price_since_entry = 0
                initial_stop_loss = 0
                atr_trailing_stop = 0
                profit_trail_active = False
                trade_count += 1
        
        # Entry logic (same as before)
        elif position_shares == 0 and signal == 1 and cash >= FIXED_POSITION_SIZE:
            position_cost = FIXED_POSITION_SIZE * (1 + TRANSACTION_COST)
            
            if cash >= position_cost and current_atr > 0:
                position_shares = FIXED_POSITION_SIZE / current_price
                entry_price = current_price
                entry_date = current_date
                highest_price_since_entry = current_price
                initial_stop_loss = entry_price * (1 - INITIAL_STOP_LOSS_PCT)
                cash -= position_cost
    
    print(f"ATR Trailing Summary: {trailing_activations} activations, {trailing_exits} exits")
    return pd.DataFrame(results), cash

# ======================
# Run Enhanced 15% Target Backtests
# ======================
print("Running enhanced 15% target backtests (TOP 3 + Combined)...")

# Test top 3 strategies + combined
enhanced_ml_results, enhanced_ml_final_cash = enhanced_15pct_atr_backtest(df, 'enhanced_ml_signal')
enhanced_williams_results, enhanced_williams_final_cash = enhanced_15pct_atr_backtest(df, 'enhanced_williams_signal')
enhanced_supertrend_results, enhanced_supertrend_final_cash = enhanced_15pct_atr_backtest(df, 'enhanced_supertrend_signal')
top_combined_results, top_combined_final_cash = enhanced_15pct_atr_backtest(df, 'top_performers_combined_signal')

# ======================
# Enhanced Performance Metrics for 15% Targets
# ======================
def calculate_15pct_metrics(results_df, final_cash, strategy_name):
    """Calculate comprehensive metrics for 15% target strategies"""
    if len(results_df) == 0:
        return f"\n{strategy_name} Strategy (15% Target): No trades executed"
    
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['profit_loss'] > 0])
    win_rate = winning_trades / total_trades
    
    # Enhanced exit analysis
    exit_reasons = results_df['exit_reason'].value_counts()
    atr_trailing_stops = len(results_df[results_df['exit_reason'] == 'ATR Trailing Stop'])
    profit_targets_15 = len(results_df[results_df['exit_reason'] == 'Take Profit Target (15%)'])
    signal_exits = exit_reasons.get('Signal Exit', 0)
    stop_losses = exit_reasons.get('Initial Stop Loss', 0)
    
    total_profit = results_df['profit_loss'].sum()
    avg_return = results_df['return_pct'].mean()
    max_return = results_df['return_pct'].max()
    min_return = results_df['return_pct'].min()
    
    # New metrics for 15% analysis
    avg_max_profit = results_df['max_profit_pct'].mean()  # Average maximum profit reached
    trades_above_15 = len(results_df[results_df['max_profit_pct'] >= 15])  # How many could have hit 15%
    
    total_return = (final_cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Risk metrics
    std_return = results_df['return_pct'].std()
    sharpe = avg_return / std_return if std_return > 0 else 0
    
    # Max drawdown
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
    
    # ATR trailing effectiveness
    atr_trades = results_df[results_df['atr_trail_active'] == True]
    atr_success_rate = len(atr_trades[atr_trades['profit_loss'] > 0]) / len(atr_trades) if len(atr_trades) > 0 else 0
    
    return f"""
{strategy_name} Strategy Results (15% Target):
===========================================
TRADE SUMMARY:
Total Trades: {total_trades}
Winning Trades: {winning_trades}
Win Rate: {win_rate:.2%}

FINANCIAL PERFORMANCE:
Total P&L: Rs{total_profit:,.2f}
Total Return: {total_return:.2%}
Avg Return/Trade: {avg_return:.2f}%
Best Trade: {max_return:.2f}%
Worst Trade: {min_return:.2f}%
Final Portfolio: Rs{final_cash:,.2f}

15% TARGET ANALYSIS:
Trades Hitting 15% Target: {profit_targets_15} ({profit_targets_15/total_trades*100:.1f}%)
Trades Above 15% Peak: {trades_above_15} ({trades_above_15/total_trades*100:.1f}%)
Avg Maximum Profit: {avg_max_profit:.2f}%

RISK METRICS:
Volatility: {std_return:.2f}%
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_drawdown:.2f}%

EXIT ANALYSIS:
- ATR Trailing Stops: {atr_trailing_stops} ({atr_trailing_stops/total_trades*100:.1f}%)
- 15% Profit Targets: {profit_targets_15} ({profit_targets_15/total_trades*100:.1f}%)
- Signal Exits: {signal_exits} ({signal_exits/total_trades*100:.1f}%)
- Initial Stop Losses: {stop_losses} ({stop_losses/total_trades*100:.1f}%)

ATR TRAILING EFFECTIVENESS:
- Trades with ATR Trailing: {len(atr_trades)}
- ATR Success Rate: {atr_success_rate:.1%}
- Avg ATR Trailing Return: {atr_trades['return_pct'].mean():.2f}%
"""

# Print enhanced 15% results
print("\n" + "="*70)
print("ENHANCED STRATEGY BACKTEST RESULTS (15% PROFIT TARGET)")
print("="*70)

print(calculate_15pct_metrics(enhanced_ml_results, enhanced_ml_final_cash, "Enhanced ML"))
print(calculate_15pct_metrics(enhanced_williams_results, enhanced_williams_final_cash, "Enhanced Williams %R"))
print(calculate_15pct_metrics(enhanced_supertrend_results, enhanced_supertrend_final_cash, "Enhanced SuperTrend"))
print(calculate_15pct_metrics(top_combined_results, top_combined_final_cash, "Top Performers Combined"))

# ======================
# 10% vs 15% Comparison Summary
# ======================
print("\n" + "="*70)
print("10% vs 15% TARGET COMPARISON SUMMARY")
print("="*70)

strategies_15_data = {
    'Enhanced ML': {'final': enhanced_ml_final_cash, 'trades': len(enhanced_ml_results), 'win_rate': len(enhanced_ml_results[enhanced_ml_results['profit_loss'] > 0])/len(enhanced_ml_results) if len(enhanced_ml_results) > 0 else 0},
    'Enhanced Williams %R': {'final': enhanced_williams_final_cash, 'trades': len(enhanced_williams_results), 'win_rate': len(enhanced_williams_results[enhanced_williams_results['profit_loss'] > 0])/len(enhanced_williams_results) if len(enhanced_williams_results) > 0 else 0},
    'Enhanced SuperTrend': {'final': enhanced_supertrend_final_cash, 'trades': len(enhanced_supertrend_results), 'win_rate': len(enhanced_supertrend_results[enhanced_supertrend_results['profit_loss'] > 0])/len(enhanced_supertrend_results) if len(enhanced_supertrend_results) > 0 else 0},
    'Top Combined': {'final': top_combined_final_cash, 'trades': len(top_combined_results), 'win_rate': len(top_combined_results[top_combined_results['profit_loss'] > 0])/len(top_combined_results) if len(top_combined_results) > 0 else 0}
}

print("STRATEGY RANKING (15% Targets):")
ranked_15_strategies = sorted(strategies_15_data.items(), key=lambda x: x[1]['final'], reverse=True)
for i, (strategy, stats) in enumerate(ranked_15_strategies, 1):
    return_pct = (stats['final'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"{i}. {strategy}: Rs{stats['final']:,.0f} ({return_pct:.1f}% return, {stats['win_rate']:.1%} win rate, {stats['trades']} trades)")

# ======================
# Save Enhanced Results
# ======================
if len(enhanced_ml_results) > 0:
    enhanced_ml_results.to_csv('enhanced_ml_15pct_backtest.csv', index=False)

if len(enhanced_williams_results) > 0:
    enhanced_williams_results.to_csv('enhanced_williams_15pct_backtest.csv', index=False)
    
if len(enhanced_supertrend_results) > 0:
    enhanced_supertrend_results.to_csv('enhanced_supertrend_15pct_backtest.csv', index=False)

if len(top_combined_results) > 0:
    top_combined_results.to_csv('top_combined_15pct_backtest.csv', index=False)

print(f"\nENHANCED 15% TARGET BACKTESTING COMPLETE!")
print("Key Enhanced Features:")
print("- 15% profit targets (increased from 10%)")
print("- 8% ATR trailing trigger (lowered from 6%)")
print("- 2.0x ATR multiplier (tightened from 2.5x)")
print("- Higher ML confidence threshold (70%)")
print("- Enhanced Williams %R with momentum filters")
print("- Improved SuperTrend with momentum confirmation")
print("- Top performers combined strategy")
print("- Detailed 15% target achievement analysis")
print("- Enhanced exit reason tracking")
print("- Maximum profit reached per trade analysis")
