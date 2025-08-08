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
    'ml_ultra_tight_atr': {'target': 0.15, 'trail_trigger': 0.05, 'atr_mult': 0.75},
    'williams_optimized': {'target': 0.10, 'trail_trigger': 0.04, 'atr_mult': 0.75},
    'ml_tiered_exit': {'target': 0.15, 'trail_trigger': 0.05, 'atr_mult': 1.0},
    'ml_adaptive_trail': {'target': 0.15, 'trail_trigger': 0.04, 'atr_mult': 'dynamic'},
    'williams_tight_trail': {'target': 0.12, 'trail_trigger': 0.04, 'atr_mult': 0.5}
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

print(f"Loaded {len(df)} rows of data")

# Data cleaning with proper index management
initial_rows = len(df)
df = df[
    (df['close'] >= MIN_PRICE) & 
    (df['close'] <= MAX_PRICE) & 
    (df['close'].notna()) &
    (np.isfinite(df['close']))
].copy()

df['price_change'] = df['close'].pct_change().abs()
df = df[df['price_change'] < 0.5].copy()

# IMPORTANT: Reset index after filtering
df = df.reset_index(drop=True)

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

# Fixed Parabolic SAR calculation with proper indexing
def calculate_parabolic_sar(df, af_start=0.02, af_increment=0.02, af_max=0.2):
    """Calculate Parabolic SAR with proper DataFrame indexing"""
    sar = df['close'].copy()
    trend = pd.Series([1] * len(df), index=df.index)  # 1 for uptrend, -1 for downtrend
    af = pd.Series([af_start] * len(df), index=df.index)
    ep = df['close'].copy()  # Extreme point
    
    for i in range(1, len(df)):
        prev_sar = sar.iloc[i-1]
        prev_trend = trend.iloc[i-1]
        prev_af = af.iloc[i-1]
        prev_ep = ep.iloc[i-1]
        
        current_high = df['close'].iloc[i]
        current_low = df['close'].iloc[i]
        
        if prev_trend == 1:  # Uptrend
            sar.iloc[i] = prev_sar + prev_af * (prev_ep - prev_sar)
            
            if current_low <= sar.iloc[i]:
                # Trend reversal
                trend.iloc[i] = -1
                sar.iloc[i] = prev_ep
                ep.iloc[i] = current_low
                af.iloc[i] = af_start
            else:
                trend.iloc[i] = 1
                if current_high > prev_ep:
                    ep.iloc[i] = current_high
                    af.iloc[i] = min(af_max, prev_af + af_increment)
                else:
                    ep.iloc[i] = prev_ep
                    af.iloc[i] = prev_af
        else:  # Downtrend
            sar.iloc[i] = prev_sar - prev_af * (prev_sar - prev_ep)
            
            if current_high >= sar.iloc[i]:
                # Trend reversal
                trend.iloc[i] = 1
                sar.iloc[i] = prev_ep
                ep.iloc[i] = current_high
                af.iloc[i] = af_start
            else:
                trend.iloc[i] = -1
                if current_low < prev_ep:
                    ep.iloc[i] = current_low
                    af.iloc[i] = min(af_max, prev_af + af_increment)
                else:
                    ep.iloc[i] = prev_ep
                    af.iloc[i] = prev_af
    
    return sar

# Calculate SAR safely
df['sar'] = calculate_parabolic_sar(df)

# Williams %R (our proven RSI alternative)
df['williams_r'] = ((df['close'].rolling(14).max() - df['close']) / 
                   (df['close'].rolling(14).max() - df['close'].rolling(14).min())) * -100

print(f"Enhanced features available: {available_features}")
df = df.dropna(subset=available_features + ['atr', 'williams_r']).reset_index(drop=True)

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
    stage1_exits = 0  # 8% exits
    stage2_exits = 0  # 12% exits
    
    print(f"\nTesting {exit_strategy_name}: Target={take_profit_pct*100}%, Trigger={trail_trigger*100}%, ATR={atr_multiplier}")
    
    for i in range(1, len(df)):
        if trade_count >= max_trades:
            break
            
        current_date = df.loc[i, 'date'] if 'date' in df.columns else i
        current_price = df.loc[i, 'close']
        current_atr = df.loc[i, 'atr']
        current_volatility = df.loc[i, 'volatility'] if not pd.isna(df.loc[i, 'volatility']) else 0.02
        signal = df.loc[i, signal_column]
        
        if current_price <= 0 or not np.isfinite(current_price) or pd.isna(current_atr):
            continue
        
        # Enhanced exit logic with multiple strategies
        if position_shares > 0 and entry_price > 0:
            if current_price > highest_price_since_entry:
                highest_price_since_entry = current_price
            
            pct_change = (current_price - entry_price) / entry_price
            
            # Dynamic ATR multiplier for adaptive strategy
            if atr_multiplier == 'dynamic':
                # Adaptive multiplier based on volatility (0.3x to 1.5x range)
                dynamic_atr_mult = max(0.3, min(1.5, current_volatility * 50))
            else:
                dynamic_atr_mult = atr_multiplier
            
            # Activate trailing stops
            if not profit_trail_active and pct_change >= trail_trigger:
                profit_trail_active = True
                trailing_activations += 1
                atr_trailing_stop = current_price - (current_atr * dynamic_atr_mult)
                print(f"Trail activated: Entry Rs{entry_price:.2f}, Current Rs{current_price:.2f}, ATR Stop Rs{atr_trailing_stop:.2f}")
            
            # Update trailing stops (only moves up)
            if profit_trail_active:
                new_atr_stop = current_price - (current_atr * dynamic_atr_mult)
                if new_atr_stop > atr_trailing_stop:
                    atr_trailing_stop = new_atr_stop
            
            should_exit = False
            exit_reason = ""
            
            # Exit condition logic based on strategy
            if current_price <= initial_stop_loss:
                should_exit = True
                exit_reason = "Initial Stop Loss"
                
            elif exit_strategy_name == 'ml_tiered_exit':
                # Tiered profit taking for ML strategy
                if 0.08 <= pct_change < 0.12:
                    should_exit = True
                    exit_reason = "Tiered Exit Stage 1 (8%)"
                    stage1_exits += 1
                elif 0.12 <= pct_change < 0.18:
                    should_exit = True
                    exit_reason = "Tiered Exit Stage 2 (12%)"
                    stage2_exits += 1
                elif profit_trail_active and current_price <= atr_trailing_stop:
                    should_exit = True
                    exit_reason = "ATR Trailing Stop"
                    trailing_exits += 1
                elif pct_change >= take_profit_pct:
                    should_exit = True
                    exit_reason = "Final Target"
                    
            elif profit_trail_active and current_price <= atr_trailing_stop:
                should_exit = True
                exit_reason = "ATR Trailing Stop"
                trailing_exits += 1
                print(f"ATR trailing exit: Rs{current_price:.2f} <= Rs{atr_trailing_stop:.2f}")
                
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
                    'atr_stop_price': atr_trailing_stop if profit_trail_active else 0,
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
    
    print(f"Results: {trailing_activations} trail activations, {trailing_exits} trail exits, Stage1: {stage1_exits}, Stage2: {stage2_exits}")
    return pd.DataFrame(results), cash

# ======================
# Run Advanced Exit Strategy Tests
# ======================
print("Running comprehensive exit strategy comparison...")

test_results = {}

# ML Strategy with ultra-tight ATR (0.75x)
ml_ultra_results, ml_ultra_cash = advanced_exit_strategy_backtest(
    df, 'enhanced_ml_signal', 'ml_ultra_tight_atr', EXIT_STRATEGIES['ml_ultra_tight_atr']
)
test_results['ML Ultra-Tight ATR (0.75x)'] = {'results': ml_ultra_results, 'cash': ml_ultra_cash}

# ML Strategy with tiered exits
ml_tiered_results, ml_tiered_cash = advanced_exit_strategy_backtest(
    df, 'enhanced_ml_signal', 'ml_tiered_exit', EXIT_STRATEGIES['ml_tiered_exit']
)
test_results['ML Tiered Exit (8%/12%/15%)'] = {'results': ml_tiered_results, 'cash': ml_tiered_cash}

# ML Strategy with adaptive trailing
ml_adaptive_results, ml_adaptive_cash = advanced_exit_strategy_backtest(
    df, 'enhanced_ml_signal', 'ml_adaptive_trail', EXIT_STRATEGIES['ml_adaptive_trail']
)
test_results['ML Adaptive Trailing'] = {'results': ml_adaptive_results, 'cash': ml_adaptive_cash}

# Williams %R with optimized parameters
williams_opt_results, williams_opt_cash = advanced_exit_strategy_backtest(
    df, 'enhanced_williams_signal', 'williams_optimized', EXIT_STRATEGIES['williams_optimized']
)
test_results['Williams %R Optimized'] = {'results': williams_opt_results, 'cash': williams_opt_cash}

# Williams %R with ultra-tight trailing
williams_tight_results, williams_tight_cash = advanced_exit_strategy_backtest(
    df, 'enhanced_williams_signal', 'williams_tight_trail', EXIT_STRATEGIES['williams_tight_trail']
)
test_results['Williams %R Ultra-Tight'] = {'results': williams_tight_results, 'cash': williams_tight_cash}

# ======================
# Advanced Performance Analysis
# ======================
def calculate_exit_strategy_metrics(results_df, final_cash, strategy_name):
    """Calculate comprehensive exit strategy performance metrics"""
    if len(results_df) == 0:
        return f"\n{strategy_name}: No trades executed"
    
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['profit_loss'] > 0])
    win_rate = winning_trades / total_trades
    
    # Detailed exit analysis
    exit_reasons = results_df['exit_reason'].value_counts()
    atr_trailing_exits = len(results_df[results_df['exit_reason'] == 'ATR Trailing Stop'])
    tiered_exits = len(results_df[results_df['exit_reason'].str.contains('Tiered', na=False)])
    profit_targets = len(results_df[results_df['exit_reason'].str.contains('Target', na=False)])
    
    total_profit = results_df['profit_loss'].sum()
    avg_return = results_df['return_pct'].mean()
    max_return = results_df['return_pct'].max()
    min_return = results_df['return_pct'].min()
    avg_max_profit = results_df['max_profit_pct'].mean()
    
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
    
    # ATR trailing effectiveness
    atr_trail_trades = results_df[results_df['trail_active'] == True]
    atr_success_rate = len(atr_trail_trades[atr_trail_trades['profit_loss'] > 0]) / len(atr_trail_trades) if len(atr_trail_trades) > 0 else 0
    
    # Average ATR multiplier used (for adaptive strategies)
    avg_atr_mult = results_df['atr_mult_used'].mean() if 'atr_mult_used' in results_df.columns else 0
    
    return f"""
{strategy_name} Advanced Exit Results:
===========================================
PERFORMANCE SUMMARY:
Total Trades: {total_trades} | Win Rate: {win_rate:.2%}
Total Return: {total_return:.2%} | Final: Rs{final_cash:,.0f}
Avg Return/Trade: {avg_return:.2f}% | Max: {max_return:.2f}% | Min: {min_return:.2f}%
Avg Max Profit Reached: {avg_max_profit:.2f}%
Sharpe Ratio: {sharpe:.2f} | Max Drawdown: {max_drawdown:.2f}%

EXIT BREAKDOWN:
• ATR Trailing Stops: {atr_trailing_exits} ({atr_trailing_exits/total_trades*100:.1f}%)
• Tiered Exits: {tiered_exits} ({tiered_exits/total_trades*100:.1f}%)
• Profit Targets: {profit_targets} ({profit_targets/total_trades*100:.1f}%)
• Stop Losses: {exit_reasons.get('Initial Stop Loss', 0)} ({exit_reasons.get('Initial Stop Loss', 0)/total_trades*100:.1f}%)

ATR TRAILING ANALYSIS:
• Trades with Trailing: {len(atr_trail_trades)} | Success Rate: {atr_success_rate:.1%}
• Avg ATR Multiplier: {avg_atr_mult:.2f}x
• Avg Trailing Return: {atr_trail_trades['return_pct'].mean():.2f}%
"""

# Print comprehensive results
print("\n" + "="*80)
print("ADVANCED EXIT STRATEGY PERFORMANCE COMPARISON")
print("="*80)

for strategy_name, data in test_results.items():
    print(calculate_exit_strategy_metrics(data['results'], data['cash'], strategy_name))

# ======================
# Final Rankings and Recommendations
# ======================
print("\n" + "="*80)
print("FINAL EXIT STRATEGY RANKINGS")
print("="*80)

# Create ranking data
ranking_data = []
for name, data in test_results.items():
    if len(data['results']) > 0:
        results_df = data['results']
        win_rate = len(results_df[results_df['profit_loss'] > 0]) / len(results_df)
        total_return = (data['cash'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        atr_exits = len(results_df[results_df['exit_reason'] == 'ATR Trailing Stop'])
        avg_return = results_df['return_pct'].mean()
        sharpe = avg_return / results_df['return_pct'].std() if results_df['return_pct'].std() > 0 else 0
        
        ranking_data.append({
            'name': name,
            'final_cash': data['cash'],
            'return_pct': total_return,
            'win_rate': win_rate,
            'trades': len(results_df),
            'atr_exits': atr_exits,
            'sharpe': sharpe
        })

# Sort by total return
ranking_data.sort(key=lambda x: x['return_pct'], reverse=True)

print("BEST PERFORMING EXIT STRATEGIES:")
for i, strategy in enumerate(ranking_data, 1):
    print(f"{i}. {strategy['name']}")
    print(f"   Portfolio: Rs{strategy['final_cash']:,.0f} ({strategy['return_pct']:.1f}% return)")
    print(f"   Win Rate: {strategy['win_rate']:.1%} | ATR Exits: {strategy['atr_exits']} | Sharpe: {strategy['sharpe']:.2f}")
    print()

# Identify the winner
if ranking_data:
    winner = ranking_data[0]
    print(f"🏆 RECOMMENDED STRATEGY: {winner['name']}")
    print(f"   Best overall performance with {winner['return_pct']:.1f}% returns")
    print(f"   {winner['win_rate']:.1%} win rate and {winner['atr_exits']} ATR trailing exits")

# ======================
# Save Results
# ======================
for strategy_name, data in test_results.items():
    if len(data['results']) > 0:
        filename = f"{strategy_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_results.csv"
        data['results'].to_csv(filename, index=False)

print(f"\nADVANCED EXIT STRATEGY TESTING COMPLETE!")
print("\nKey improvements tested:")
print("- 0.75x ultra-tight ATR trailing stops")
print("- Tiered profit taking (8%/12%/15%)")
print("- Adaptive ATR multipliers")
print("- Optimized Williams %R parameters")
print("- Comprehensive exit reason tracking")
print("- Fixed indexing issues")
print("- All results saved to CSV files")
