import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================
# Enhanced Strategy Configurations
# ======================
CSV_PATH = "your_training_data.csv"
MODEL_PATH = "model.pkl"

# Trading parameters (in Indian Rupees)
INITIAL_CAPITAL = 1000000  # ₹10 Lakhs starting capital
FIXED_POSITION_SIZE = 100000  # ₹1 Lakh per trade
INITIAL_STOP_LOSS_PCT = 0.05  # 5% initial stop loss
TAKE_PROFIT_PCT = 0.10  # 10% take profit target (starting with 10%)
PROFIT_TRAIL_TRIGGER = 0.06  # Start trailing after 6% profit
ATR_MULTIPLIER = 2.5  # ATR multiplier for dynamic trailing
MIN_PRICE = 50.0
MAX_PRICE = 10000.0
TRANSACTION_COST = 0.001

# ======================
# Load and Clean Data
# ======================
print("Loading data and model for enhanced strategies...")
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
# Enhanced Technical Indicators
# ======================
features = ['rsi', 'atr', 'adx', 'ema10', 'ema21', 'volumechange']
available_features = [f for f in features if f in df.columns]

# Calculate ATR for dynamic trailing stops (estimate from close if OHLCV not available)
if 'atr' not in df.columns:
    df['atr'] = df['close'].rolling(14).apply(lambda x: x.std() * 1.5)

# STRATEGY 1: BOLLINGER BANDS MEAN REVERSION
bb_data = ta.bbands(df['close'], length=20, std=2)
df['bb_upper'] = bb_data['BBU_20_2.0']
df['bb_lower'] = bb_data['BBL_20_2.0']
df['bb_middle'] = bb_data['BBM_20_2.0']

# STRATEGY 2: SUPERTREND (using close price approximation)
df['hl2'] = (df['close'] + df['close'].shift(1)) / 2  # Approximation for HL2
supertrend_data = ta.supertrend(df['close'], df['close'], df['close'], length=10, multiplier=3)
if supertrend_data is not None:
    df['supertrend'] = supertrend_data['SUPERT_10_3.0']
    df['supertrend_direction'] = supertrend_data['SUPERTd_10_3.0']
else:
    df['supertrend'] = df['close']
    df['supertrend_direction'] = 1

# STRATEGY 3: DONCHIAN CHANNELS  
df['donchian_upper'] = df['close'].rolling(20).max()
df['donchian_lower'] = df['close'].rolling(20).min()
df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2

# STRATEGY 4: WILLIAMS %R (approximation using close prices)
df['williams_r'] = ((df['close'].rolling(14).max() - df['close']) / 
                   (df['close'].rolling(14).max() - df['close'].rolling(14).min())) * -100

print(f"Enhanced features available: {available_features}")
df = df.dropna(subset=available_features + ['atr', 'bb_upper']).reset_index(drop=True)

# ======================
# Strategy Signal Generation
# ======================
print("Generating enhanced strategy signals...")
X = df[available_features]

df['ml_signal'] = model.predict(X)
df['ml_probability'] = model.predict_proba(X)[:, 1]

# BOLLINGER BANDS MEAN REVERSION STRATEGY
df['bb_signal'] = 0
bb_buy_condition = (
    (df['close'] < df['bb_lower']) &  # Price below lower band
    (df['close'] > df['close'].shift(1)) &  # Price starting to recover
    (df['close'] > df['bb_middle'].shift(5))  # Still above medium-term average
)
df.loc[bb_buy_condition, 'bb_signal'] = 1

bb_sell_condition = (
    (df['close'] > df['bb_upper']) |  # Overbought
    (df['close'] < df['bb_middle'])  # Back to mean
)
df.loc[bb_sell_condition, 'bb_signal'] = -1

# SUPERTREND STRATEGY
df['supertrend_signal'] = 0
if 'supertrend_direction' in df.columns:
    df.loc[df['supertrend_direction'] == 1, 'supertrend_signal'] = 1
    df.loc[df['supertrend_direction'] == -1, 'supertrend_signal'] = -1

# DONCHIAN BREAKOUT STRATEGY
df['donchian_signal'] = 0
donchian_buy_condition = (
    (df['close'] > df['donchian_upper'].shift(1)) &  # Breakout above upper channel
    (df['close'] > df['close'].shift(5))  # Momentum confirmation
)
df.loc[donchian_buy_condition, 'donchian_signal'] = 1

donchian_sell_condition = (df['close'] < df['donchian_lower'].shift(1))
df.loc[donchian_sell_condition, 'donchian_signal'] = -1

# WILLIAMS %R STRATEGY
df['williams_signal'] = 0
williams_buy_condition = (
    (df['williams_r'] < -80) &  # Oversold
    (df['williams_r'] > df['williams_r'].shift(1))  # Starting to turn up
)
df.loc[williams_buy_condition, 'williams_signal'] = 1

williams_sell_condition = (df['williams_r'] > -20)  # Overbought
df.loc[williams_sell_condition, 'williams_signal'] = -1

# ENHANCED COMBINED STRATEGY
df['enhanced_combined_signal'] = 0
combined_buy_condition = (
    (df['ml_signal'] == 1) & 
    (df['ml_probability'] > 0.65) &
    (
        (df['bb_signal'] == 1) |  # BB mean reversion
        (df['supertrend_signal'] == 1) |  # Supertrend bullish
        (df['donchian_signal'] == 1)  # Donchian breakout
    )
)
df.loc[combined_buy_condition, 'enhanced_combined_signal'] = 1

combined_sell_condition = (
    (df['bb_signal'] == -1) |
    (df['supertrend_signal'] == -1) |
    (df['donchian_signal'] == -1) |
    (df['williams_signal'] == -1)
)
df.loc[combined_sell_condition, 'enhanced_combined_signal'] = -1

# ======================
# Advanced ATR-Based Trailing Stop Engine
# ======================
def advanced_atr_trailing_backtest(df, signal_column, initial_capital=INITIAL_CAPITAL):
    """
    Advanced backtesting with ATR-based dynamic trailing stops
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
    
    for i in range(1, len(df)):
        if trade_count >= max_trades:
            break
            
        current_date = df.loc[i, 'date'] if 'date' in df.columns else i
        current_price = df.loc[i, 'close']
        current_atr = df.loc[i, 'atr']
        signal = df.loc[i, signal_column]
        
        if current_price <= 0 or not np.isfinite(current_price) or pd.isna(current_atr):
            continue
        
        # Exit logic with ATR-based trailing stops
        if position_shares > 0 and entry_price > 0:
            if current_price > highest_price_since_entry:
                highest_price_since_entry = current_price
            
            pct_change = (current_price - entry_price) / entry_price
            
            # Activate ATR-based trailing after profit trigger
            if not profit_trail_active and pct_change >= PROFIT_TRAIL_TRIGGER:
                profit_trail_active = True
                atr_trailing_stop = current_price - (current_atr * ATR_MULTIPLIER)
            
            # Update ATR trailing stop (only moves up)
            if profit_trail_active:
                new_atr_stop = current_price - (current_atr * ATR_MULTIPLIER)
                if new_atr_stop > atr_trailing_stop:
                    atr_trailing_stop = new_atr_stop
            
            should_exit = False
            exit_reason = ""
            
            # Exit conditions
            if current_price <= initial_stop_loss:
                should_exit = True
                exit_reason = "Initial Stop Loss"
            elif profit_trail_active and current_price <= atr_trailing_stop:
                should_exit = True
                exit_reason = "ATR Trailing Stop"
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
                    'atr_trail_active': profit_trail_active,
                    'atr_trail_price': atr_trailing_stop if profit_trail_active else 0,
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
    
    return pd.DataFrame(results), cash

# ======================
# Run Enhanced Backtests
# ======================
print("Running enhanced backtests with ATR trailing stops...")

# Test all strategies with 10% profit targets
ml_results, ml_final_cash = advanced_atr_trailing_backtest(df, 'ml_signal')
bb_results, bb_final_cash = advanced_atr_trailing_backtest(df, 'bb_signal')
supertrend_results, supertrend_final_cash = advanced_atr_trailing_backtest(df, 'supertrend_signal')
donchian_results, donchian_final_cash = advanced_atr_trailing_backtest(df, 'donchian_signal')
williams_results, williams_final_cash = advanced_atr_trailing_backtest(df, 'williams_signal')
enhanced_combined_results, enhanced_combined_final_cash = advanced_atr_trailing_backtest(df, 'enhanced_combined_signal')

# ======================
# Enhanced Performance Metrics (No Emoji)
# ======================
def calculate_enhanced_metrics(results_df, final_cash, strategy_name):
    """Calculate comprehensive metrics without emoji characters"""
    if len(results_df) == 0:
        return f"\n{strategy_name} Strategy: No trades executed"
    
    total_trades = len(results_df)
    winning_trades = len(results_df[results_df['profit_loss'] > 0])
    win_rate = winning_trades / total_trades
    
    # Detailed exit analysis
    exit_reasons = results_df['exit_reason'].value_counts()
    atr_trailing_stops = len(results_df[results_df['exit_reason'] == 'ATR Trailing Stop'])
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
    
    # ATR trailing effectiveness
    atr_trades = results_df[results_df['atr_trail_active'] == True]
    atr_effectiveness = len(atr_trades[atr_trades['profit_loss'] > 0]) / len(atr_trades) if len(atr_trades) > 0 else 0
    
    return f"""
{strategy_name} Strategy Results (10% Target):
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

RISK METRICS:
Volatility: {std_return:.2f}%
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_drawdown:.2f}%

EXIT ANALYSIS:
- ATR Trailing Stops: {atr_trailing_stops} ({atr_trailing_stops/total_trades*100:.1f}%)
- Profit Targets: {profit_targets} ({profit_targets/total_trades*100:.1f}%)
- Signal Exits: {signal_exits} ({signal_exits/total_trades*100:.1f}%)
- Initial Stop Losses: {stop_losses} ({stop_losses/total_trades*100:.1f}%)

ATR TRAILING EFFECTIVENESS:
- Trades with ATR Trailing: {len(atr_trades)}
- ATR Success Rate: {atr_effectiveness:.1%}
- Avg ATR Trailing Return: {atr_trades['return_pct'].mean():.2f}%
"""

# Print enhanced results
print("\n" + "="*70)
print("ENHANCED STRATEGY BACKTEST RESULTS (10% PROFIT TARGET)")
print("="*70)

print(calculate_enhanced_metrics(ml_results, ml_final_cash, "ML"))
print(calculate_enhanced_metrics(bb_results, bb_final_cash, "Bollinger Bands"))
print(calculate_enhanced_metrics(supertrend_results, supertrend_final_cash, "SuperTrend"))
print(calculate_enhanced_metrics(donchian_results, donchian_final_cash, "Donchian Channels"))
print(calculate_enhanced_metrics(williams_results, williams_final_cash, "Williams %R"))
print(calculate_enhanced_metrics(enhanced_combined_results, enhanced_combined_final_cash, "Enhanced Combined"))

# ======================
# Performance Ranking
# ======================
strategies_data = {
    'ML': {'trades': len(ml_results), 'win_rate': len(ml_results[ml_results['profit_loss'] > 0])/len(ml_results) if len(ml_results) > 0 else 0, 'final': ml_final_cash},
    'Bollinger Bands': {'trades': len(bb_results), 'win_rate': len(bb_results[bb_results['profit_loss'] > 0])/len(bb_results) if len(bb_results) > 0 else 0, 'final': bb_final_cash},
    'SuperTrend': {'trades': len(supertrend_results), 'win_rate': len(supertrend_results[supertrend_results['profit_loss'] > 0])/len(supertrend_results) if len(supertrend_results) > 0 else 0, 'final': supertrend_final_cash},
    'Donchian': {'trades': len(donchian_results), 'win_rate': len(donchian_results[donchian_results['profit_loss'] > 0])/len(donchian_results) if len(donchian_results) > 0 else 0, 'final': donchian_final_cash},
    'Williams %R': {'trades': len(williams_results), 'win_rate': len(williams_results[williams_results['profit_loss'] > 0])/len(williams_results) if len(williams_results) > 0 else 0, 'final': williams_final_cash},
    'Enhanced Combined': {'trades': len(enhanced_combined_results), 'win_rate': len(enhanced_combined_results[enhanced_combined_results['profit_loss'] > 0])/len(enhanced_combined_results) if len(enhanced_combined_results) > 0 else 0, 'final': enhanced_combined_final_cash}
}

print(f"\nSTRATEGY RANKING (10% Targets):")
ranked_strategies = sorted(strategies_data.items(), key=lambda x: x[1]['final'], reverse=True)
for i, (strategy, stats) in enumerate(ranked_strategies, 1):
    return_pct = (stats['final'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"{i}. {strategy}: Rs{stats['final']:,.0f} ({return_pct:.1f}% return, {stats['win_rate']:.1%} win rate, {stats['trades']} trades)")

# ======================
# Save Results
# ======================
if len(ml_results) > 0:
    ml_results.to_csv('ml_enhanced_10pct_backtest.csv', index=False)

if len(bb_results) > 0:
    bb_results.to_csv('bollinger_bands_10pct_backtest.csv', index=False)
    
if len(supertrend_results) > 0:
    supertrend_results.to_csv('supertrend_10pct_backtest.csv', index=False)

if len(donchian_results) > 0:
    donchian_results.to_csv('donchian_10pct_backtest.csv', index=False)

if len(williams_results) > 0:
    williams_results.to_csv('williams_r_10pct_backtest.csv', index=False)

if len(enhanced_combined_results) > 0:
    enhanced_combined_results.to_csv('enhanced_combined_10pct_backtest.csv', index=False)

print(f"\nENHANCED BACKTESTING COMPLETE!")
print("Key Features:")
print("- ATR-based trailing stops (2.5x multiplier)")
print("- 10% take profit targets")
print("- 6 different strategies tested")
print("- Dynamic volatility adaptation")
print("- All results saved with '10pct' suffix")
print("\nNext: Modify TAKE_PROFIT_PCT to 0.15 for 15% comparison")
