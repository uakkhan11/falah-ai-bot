import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================
# Advanced Exit Strategy Testing Configurations
# ======================
CSV_PATH = "your_training_data.csv"
MODEL_PATH = "model.pkl"

# Base trading parameters
INITIAL_CAPITAL = 1000000  # ₹10 Lakhs starting capital
FIXED_POSITION_SIZE = 100000  # ₹1 Lakh per trade
INITIAL_STOP_LOSS_PCT = 0.05  # 5% initial stop loss
TRANSACTION_COST = 0.001

# Different exit strategies to test
EXIT_STRATEGIES = {
    'tiered_ml': {'target': 0.15, 'trail_trigger': 0.05, 'atr_mult': 0.75},
    'williams_optimized': {'target': 0.10, 'trail_trigger': 0.04, 'atr_mult': 0.75},
    'chandelier_exit': {'target': 0.15, 'trail_trigger': 0.06, 'atr_mult': 1.0},
    'parabolic_sar': {'target': 0.12, 'trail_trigger': 0.05, 'atr_mult': 0.5},
    'volatility_adaptive': {'target': 0.15, 'trail_trigger': 0.05, 'atr_mult': 'dynamic'}
}

MIN_PRICE = 50.0
MAX_PRICE = 10000.0

# ======================
# Load and Clean Data
# ======================
print("Loading data for advanced exit strategy testing...")
df = pd.read_csv(CSV_PATH)
df.columns = [c.lower() for c in df.columns]

model = joblib.load(MODEL_PATH)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.tz_localize(None)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

print(f"Loaded {len(df)} rows of data")

# Data cleaning
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
# Enhanced Technical Indicators for Exit Strategies
# ======================
features = ['rsi', 'atr', 'adx', 'ema10', 'ema21', 'volumechange']
available_features = [f for f in features if f in df.columns]

# Calculate ATR for dynamic trailing stops
if 'atr' not in df.columns:
    df['atr'] = df['close'].rolling(14).apply(lambda x: x.std() * 1.5)

# Calculate additional exit indicators
df['atr_20'] = df['close'].rolling(20).apply(lambda x: x.std() * 1.2)  # Longer-term ATR
df['volatility'] = df['close'].pct_change().rolling(10).std()  # 10-period volatility

# Parabolic SAR approximation
df['sar'] = df['close'].copy()
for i in range(1, len(df)):
    if df.loc[i, 'close'] > df.loc[i-1, 'close']:
        df.loc[i, 'sar'] = df.loc[i-1, 'sar'] + 0.02 * (df.loc[i, 'close'] - df.loc[i-1, 'sar'])
    else:
        df.loc[i, 'sar'] = df.loc[i-1, 'sar'] - 0.02 * (df.loc[i-1, 'sar'] - df.loc[i, 'close'])

# Chandelier Exit calculation
df['chandelier_exit'] = df['close'].rolling(22).max() - (df['atr'] * 3)

# Williams %R (our proven RSI alternative)
df['williams_r'] = ((df['close'].rolling(14).max() - df['close']) / 
                   (df['close'].rolling(14).max() - df['close'].rolling(14).min())) * -100

print(f"Enhanced features available: {available_features}")
df = df.dropna(subset=available_features + ['atr', 'williams_r', 'sar']).reset_index(drop=True)

# ======================
# Strategy Signal Generation
# ======================
print("Generating signals for ML and Williams %R strategies...")
X = df[available_features]

df['ml_signal'] = model.predict(X)
df['ml_probability'] = model.predict_proba(X)[:, 1]

# Enhanced ML Strategy (70% confidence)
df['enhanced_ml_signal'] = 0
df.loc[(df['ml_signal'] == 1) & (df['ml_probability'] > 0.70), 'enhanced_ml_signal'] = 1
df.loc[(df['ml_signal'] == 0) & (df['ml_probability'] < 0.35), 'enhanced_ml_signal'] = -1

# Enhanced Williams %R Strategy
df['enhanced_williams_signal'] = 0
williams_buy_condition = (
    (df['williams_r'] < -80) &
    (df['williams_r'] > df['williams_r'].shift(1)) &
    (df['close'] > df['close'].shift(3))
)
df.loc[williams_buy_condition, 'enhanced_williams_signal'] = 1

williams_sell_condition = (df['williams_r'] > -15) | (df['close'] < df['close'].shift(5))
df.loc[williams_sell_condition, 'enhanced_williams_signal'] = -1

# ======================
# Advanced Exit Strategy Backtesting Engine
# ======================
def advanced_exit_strategy_backtest(df, signal_column, exit_strategy_name, strategy_params, initial_capital=INITIAL_CAPITAL):
    """
    Advanced backtesting with multiple exit strategy models
    """
    results = []
    cash = initial_capital
    position_shares = 0
    entry_price = 0
    entry_date = None
    highest_price_since_entry = 0
    initial_stop_loss = 0
    
    # Exit strategy variables
    atr_trailing_stop = 0
    profit_trail_active = False
    sar_trailing_stop = 0
    chandelier_trailing_stop = 0
    
    trade_count = 0
    max_trades = 200
    
    # Extract strategy parameters
    take_profit_pct = strategy_params['target']
    trail_trigger = strategy_params['trail_trigger'] 
    atr_multiplier = strategy_params['atr_mult']
    
    # Tracking variables
    trailing_activations = 0
    trailing_exits = 0
    tiered_exits = 0
    
    print(f"\nTesting {exit_strategy_name}: Target={take_profit_pct*100}%, Trigger={trail_trigger*100}%, ATR={atr_multiplier}")
    
    for i in range(1, len(df)):
        if trade_count >= max_trades:
            break
            
        current_date = df.loc[i, 'date'] if 'date' in df.columns else i
        current_price = df.loc[i, 'close']
        current_atr = df.loc[i, 'atr']
        current_volatility = df.loc[i, 'volatility']
        signal = df.loc[i, signal_column]
        
        if current_price <= 0 or not np.isfinite(current_price) or pd.isna(current_atr):
            continue
        
        # Enhanced exit logic with multiple strategies
        if position_shares > 0 and entry_price > 0:
            if current_price > highest_price_since_entry:
                highest_price_since_entry = current_price
            
            pct_change = (current_price - entry_price) / entry_price
            
            # Dynamic ATR multiplier for volatility adaptive strategy
            if atr_multiplier == 'dynamic':
                dynamic_atr_mult = max(0.5, min(2.0, current_volatility * 100))
            else:
                dynamic_atr_mult = atr_multiplier
            
            # Activate trailing stops
            if not profit_trail_active and pct_change >= trail_trigger:
                profit_trail_active = True
                trailing_activations += 1
                
                # Different trailing stop calculations
                if exit_strategy_name == 'chandelier_exit':
                    chandelier_trailing_stop = df.loc[i, 'chandelier_exit']
                elif exit_strategy_name == 'parabolic_sar':
                    sar_trailing_stop = df.loc[i, 'sar']
                else:
                    atr_trailing_stop = current_price - (current_atr * dynamic_atr_mult)
            
            # Update trailing stops
            if profit_trail_active:
                if exit_strategy_name == 'chandelier_exit':
                    new_chandelier = df.loc[i, 'chandelier_exit']
                    if new_chandelier > chandelier_trailing_stop:
                        chandelier_trailing_stop = new_chandelier
                elif exit_strategy_name == 'parabolic_sar':
                    sar_trailing_stop = df.loc[i, 'sar']
                else:
                    new_atr_stop = current_price - (current_atr * dynamic_atr_mult)
                    if new_atr_stop > atr_trailing_stop:
                        atr_trailing_stop = new_atr_stop
            
            should_exit = False
            exit_reason = ""
            
            # Exit condition logic based on strategy
            if current_price <= initial_stop_loss:
                should_exit = True
                exit_reason = "Initial Stop Loss"
                
            elif exit_strategy_name == 'tiered_ml' and profit_trail_active:
                # Tiered profit taking for ML strategy
                if pct_change >= 0.08 and pct_change < 0.12:
                    should_exit = True
                    exit_reason = "Tiered Exit (8%)"
                    tiered_exits += 1
                elif pct_change >= 0.12 and pct_change < 0.18:
                    should_exit = True
                    exit_reason = "Tiered Exit (12%)"
                    tiered_exits += 1
                elif current_price <= atr_trailing_stop:
                    should_exit = True
                    exit_reason = "ATR Trailing Stop"
                    trailing_exits += 1
                elif pct_change >= take_profit_pct:
                    should_exit = True
                    exit_reason = "Final Target (15%)"
                    
            elif exit_strategy_name == 'chandelier_exit' and profit_trail_active:
                if current_price <= chandelier_trailing_stop:
                    should_exit = True
                    exit_reason = "Chandelier Exit"
                    trailing_exits += 1
                elif pct_change >= take_profit_pct:
                    should_exit = True
                    exit_reason = "Profit Target"
                    
            elif exit_strategy_name == 'parabolic_sar' and profit_trail_active:
                if current_price <= sar_trailing_stop:
                    should_exit = True
                    exit_reason = "SAR Trailing Stop"
                    trailing_exits += 1
                elif pct_change >= take_profit_pct:
                    should_exit = True
                    exit_reason = "Profit Target"
                    
            elif profit_trail_active and current_price <= atr_trailing_stop:
                should_exit = True
                exit_reason = "ATR Trailing Stop"
                trailing_exits += 1
                
            elif pct_change >= take_profit_pct:
                should_exit = True
                exit_reason = "Profit Target"
                
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
                    'exit_strategy': exit_strategy_name,
                    'trail_active': profit_trail_active,
                    'atr_mult_used': dynamic_atr_mult,
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
                sar_trailing_stop = 0
                chandelier_trailing_stop = 0
                profit_trail_active = False
                trade_count += 1
        
        # Entry logic
        elif position_shares == 0 and signal == 1 and cash >= FIXED_POSITION_SIZE:
            position_cost = FIXED_POSITION_SIZE * (1 + TRANSACTION_COST)
            
            if cash >= position_cost and current_atr > 0:
                position_shares = FIXED_POSITION_SIZE / current_price
                entry_price = current_price
                entry_date = current_date
                highest_price_since_entry = current_price
                initial_stop_loss = entry_price * (1 - INITIAL_STOP_LOSS_PCT)
                cash -= position_cost
    
    print(f"{exit_strategy_name} Summary: {trailing_activations} trail activations, {trailing_exits} trail exits, {tiered_exits} tiered exits")
    return pd.DataFrame(results), cash

# ======================
# Run Advanced Exit Strategy Tests
# ======================
print("Running comprehensive exit strategy comparison...")

# Test different exit strategies on ML and Williams %R
test_results = {}

# ML Strategy with different exit approaches
ml_tiered_results, ml_tiered_cash = advanced_exit_strategy_backtest(
    df, 'enhanced_ml_signal', 'tiered_ml', EXIT_STRATEGIES['tiered_ml']
)
test_results['ML Tiered (0.75 ATR)'] = {'results': ml_tiered_results, 'cash': ml_tiered_cash}

ml_chandelier_results, ml_chandelier_cash = advanced_exit_strategy_backtest(
    df, 'enhanced_ml_signal', 'chandelier_exit', EXIT_STRATEGIES['chandelier_exit']
)
test_results['ML Chandelier Exit'] = {'results': ml_chandelier_results, 'cash': ml_chandelier_cash}

ml_sar_results, ml_sar_cash = advanced_exit_strategy_backtest(
    df, 'enhanced_ml_signal', 'parabolic_sar', EXIT_STRATEGIES['parabolic_sar']
)
test_results['ML Parabolic SAR'] = {'results': ml_sar_results, 'cash': ml_sar_cash}

ml_adaptive_results, ml_adaptive_cash = advanced_exit_strategy_backtest(
    df, 'enhanced_ml_signal', 'volatility_adaptive', EXIT_STRATEGIES['volatility_adaptive']
)
test_results['ML Volatility Adaptive'] = {'results': ml_adaptive_results, 'cash': ml_adaptive_cash}

# Williams %R with optimized exit
williams_optimized_results, williams_optimized_cash = advanced_exit_strategy_backtest(
    df, 'enhanced_williams_signal', 'williams_optimized', EXIT_STRATEGIES['williams_optimized']
)
test_results['Williams %R Optimized'] = {'results': williams_optimized_results, 'cash': williams_optimized_cash}

# ======================
# Advanced Performance Analysis
# ======================
def calculate_advanced_exit_metrics(results_df, final_cash, strategy_name):
    """Calculate comprehensive exit strategy performance metrics"""
    if len(results_df) == 0:
        return f"\n{strategy_name}: No trades executed"
    
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['profit_loss'] > 0])
    win_rate = winning_trades / total_trades
    
    # Advanced exit analysis
    exit_reasons = results_df['exit_reason'].value_counts()
    trailing_stops = len(results_df[results_df['exit_reason'].str.contains('Trailing|SAR|Chandelier', na=False)])
    tiered_exits = len(results_df[results_df['exit_reason'].str.contains('Tiered', na=False)])
    profit_targets = len(results_df[results_df['exit_reason'].str.contains('Target|Final', na=False)])
    
    total_profit = results_df['profit_loss'].sum()
    avg_return = results_df['return_pct'].mean()
    max_return = results_df['return_pct'].max()
    min_return = results_df['return_pct'].min()
    avg_max_profit = results_df['max_profit_pct'].mean()
    
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
    
    # Exit effectiveness
    trail_trades = results_df[results_df['trail_active'] == True]
    trail_effectiveness = len(trail_trades[trail_trades['profit_loss'] > 0]) / len(trail_trades) if len(trail_trades) > 0 else 0
    
    return f"""
{strategy_name} Exit Strategy Results:
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
Avg Max Profit: {avg_max_profit:.2f}%
Final Portfolio: Rs{final_cash:,.2f}

RISK METRICS:
Volatility: {std_return:.2f}%
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_drawdown:.2f}%

ADVANCED EXIT ANALYSIS:
- Trailing Stops: {trailing_stops} ({trailing_stops/total_trades*100:.1f}%)
- Tiered Exits: {tiered_exits} ({tiered_exits/total_trades*100:.1f}%)
- Profit Targets: {profit_targets} ({profit_targets/total_trades*100:.1f}%)
- Stop Losses: {exit_reasons.get('Initial Stop Loss', 0)}

TRAILING EFFECTIVENESS:
- Trades with Trailing: {len(trail_trades)}
- Trail Success Rate: {trail_effectiveness:.1%}
- Avg Trail Return: {trail_trades['return_pct'].mean():.2f}%
"""

# Print comprehensive exit strategy comparison
print("\n" + "="*80)
print("COMPREHENSIVE EXIT STRATEGY PERFORMANCE COMPARISON")
print("="*80)

for strategy_name, data in test_results.items():
    print(calculate_advanced_exit_metrics(data['results'], data['cash'], strategy_name))

# ======================
# Exit Strategy Rankings
# ======================
print("\n" + "="*80)
print("EXIT STRATEGY RANKINGS")
print("="*80)

strategy_rankings = []
for name, data in test_results.items():
    if len(data['results']) > 0:
        win_rate = len(data['results'][data['results']['profit_loss'] > 0]) / len(data['results'])
        total_return = (data['cash'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        trailing_exits = len(data['results'][data['results']['exit_reason'].str.contains('Trailing|SAR|Chandelier', na=False)])
        
        strategy_rankings.append({
            'name': name,
            'final_cash': data['cash'],
            'return_pct': total_return,
            'win_rate': win_rate,
            'trades': len(data['results']),
            'trailing_exits': trailing_exits
        })

# Sort by final portfolio value
strategy_rankings.sort(key=lambda x: x['final_cash'], reverse=True)

print("FINAL RANKINGS:")
for i, strategy in enumerate(strategy_rankings, 1):
    print(f"{i}. {strategy['name']}: Rs{strategy['final_cash']:,.0f}")
    print(f"   Return: {strategy['return_pct']:.1f}% | Win Rate: {strategy['win_rate']:.1%} | Trailing Exits: {strategy['trailing_exits']}")

# ======================
# Save Results
# ======================
for strategy_name, data in test_results.items():
    if len(data['results']) > 0:
        filename = f"{strategy_name.lower().replace(' ', '_')}_exit_test.csv"
        data['results'].to_csv(filename, index=False)

print(f"\nADVANCED EXIT STRATEGY TESTING COMPLETE!")
print("Key Features Tested:")
print("- 0.75x ATR multiplier (ultra-tight trailing)")
print("- Tiered profit taking (8%, 12%, 15%)")
print("- Chandelier Exit (ATR + price extremes)")
print("- Parabolic SAR trailing")
print("- Volatility-adaptive trailing")
print("- Comprehensive exit reason analysis")
print("- All results saved with '_exit_test' suffix")

print(f"\n🎯 RECOMMENDATION: Use the TOP RANKED exit strategy for live trading!")
