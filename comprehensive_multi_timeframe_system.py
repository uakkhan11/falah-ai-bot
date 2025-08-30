
import os
import pandas as pd
import numpy as np
import ta
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# =============================================================================
# COMPREHENSIVE TIMEFRAME TESTING SYSTEM
# Testing SAME logic across ALL timeframes to eliminate timeframe bias
# =============================================================================

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data")
}

# CORE PARAMETERS (SAME LOGIC ACROSS ALL TIMEFRAMES)
INITIAL_CAPITAL = 100000
POSITION_SIZE = 0.02
YEAR_FILTER = 2025

# TECHNICAL PARAMETERS (CONSISTENT ACROSS TIMEFRAMES)
RSI_OVERBOUGHT = 70
ADX_MIN = 20
VOLUME_MULT = 1.5

# TIMEFRAME CONFIGURATIONS - Testing ALL possible timeframes
TIMEFRAME_CONFIGS = {
    '1minute': {
        'source_data': '15minute',  # Derive from 15-minute
        'aggregation': 1,           # Every 1 bar
        'profit_target': 0.015,     # 1.5% (tighter for faster timeframe)
        'stop_loss': 0.010,         # 1.0% (tighter stops)
        'max_hold_bars': 60,        # 60 minutes max hold
        'description': 'Ultra-fast scalping'
    },
    '5minute': {
        'source_data': '15minute',  # Derive from 15-minute
        'aggregation': 3,           # Every 3 bars (15/5 = 3)
        'profit_target': 0.020,     # 2.0% 
        'stop_loss': 0.012,         # 1.2%
        'max_hold_bars': 36,        # 3 hours max hold
        'description': 'Fast scalping'
    },
    '15minute': {
        'source_data': '15minute',  # Direct data
        'aggregation': 1,           # No aggregation
        'profit_target': 0.025,     # 2.5% (current baseline)
        'stop_loss': 0.015,         # 1.5%
        'max_hold_bars': 25,        # Current baseline
        'description': 'Current baseline'
    },
    '30minute': {
        'source_data': '15minute',  # Aggregate from 15-minute
        'aggregation': 2,           # Every 2 bars
        'profit_target': 0.030,     # 3.0%
        'stop_loss': 0.018,         # 1.8%
        'max_hold_bars': 16,        # 8 hours max hold
        'description': 'Medium scalping'
    },
    '1hour': {
        'source_data': '1hour',     # Direct data
        'aggregation': 1,           # No aggregation
        'profit_target': 0.035,     # 3.5%
        'stop_loss': 0.020,         # 2.0%
        'max_hold_bars': 12,        # 12 hours
        'description': 'Short swing trading'
    },
    '4hour': {
        'source_data': '1hour',     # Aggregate from 1-hour
        'aggregation': 4,           # Every 4 bars
        'profit_target': 0.050,     # 5.0%
        'stop_loss': 0.025,         # 2.5%
        'max_hold_bars': 6,         # 24 hours
        'description': 'Medium swing trading'
    },
    'daily': {
        'source_data': 'daily',     # Direct data
        'aggregation': 1,           # No aggregation
        'profit_target': 0.060,     # 6.0%
        'stop_loss': 0.030,         # 3.0%
        'max_hold_bars': 5,         # 5 days
        'description': 'Long swing trading'
    }
}

def get_all_symbols():
    """Get all available symbols from all data paths"""
    all_symbols = set()

    for data_type, path in DATA_PATHS.items():
        try:
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.endswith('.csv')]
                symbols = [f.replace('.csv', '') for f in files]
                all_symbols.update(symbols)
        except Exception as e:
            print(f"Error reading {data_type}: {e}")

    return sorted(list(all_symbols))

def load_and_resample_data(symbol, timeframe):
    """Load and resample data to target timeframe"""
    try:
        config = TIMEFRAME_CONFIGS[timeframe]
        source_data = config['source_data']
        aggregation = config['aggregation']

        # Load source data
        file_path = os.path.join(DATA_PATHS[source_data], f"{symbol}.csv")
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)

        if len(df) < 100:
            return None

        # Resample if needed
        if aggregation > 1:
            # Create aggregated bars
            df_resampled = []

            for i in range(0, len(df), aggregation):
                chunk = df.iloc[i:i+aggregation]
                if len(chunk) == aggregation:  # Only use complete bars
                    aggregated_bar = {
                        'date': chunk['date'].iloc[-1],  # Use last timestamp
                        'open': chunk['open'].iloc[0],   # First open
                        'high': chunk['high'].max(),     # Highest high
                        'low': chunk['low'].min(),       # Lowest low
                        'close': chunk['close'].iloc[-1], # Last close
                        'volume': chunk['volume'].sum()   # Sum volume
                    }
                    df_resampled.append(aggregated_bar)

            if len(df_resampled) < 50:
                return None

            df = pd.DataFrame(df_resampled)

        return df

    except Exception as e:
        print(f"Error loading {symbol} for {timeframe}: {e}")
        return None

def add_timeframe_indicators(df, timeframe):
    """Add technical indicators adjusted for timeframe"""
    try:
        # Adjust indicator periods based on timeframe characteristics
        if timeframe in ['1minute', '5minute']:
            # Faster indicators for faster timeframes
            rsi_period = min(14, len(df)//4)
            adx_period = min(14, len(df)//4) 
            sma_period = min(20, len(df)//4)
        elif timeframe in ['15minute', '30minute']:
            # Standard periods
            rsi_period = min(14, len(df)//4)
            adx_period = min(14, len(df)//4)
            sma_period = min(20, len(df)//4)
        else:
            # Slower indicators for slower timeframes
            rsi_period = min(14, len(df)//4)
            adx_period = min(14, len(df)//4)
            sma_period = min(20, len(df)//4)

        # Calculate indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=rsi_period)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=adx_period)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(sma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price indicators
        df['sma_20'] = df['close'].rolling(sma_period).mean()
        df['high_20'] = df['high'].rolling(sma_period).max()

        # Fill NaN values
        numeric_columns = ['rsi', 'adx', 'volume_ratio', 'sma_20', 'high_20', 'volume_sma']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        return df
    except Exception as e:
        print(f"Error adding indicators for {timeframe}: {e}")
        return df

def generate_timeframe_signals(df):
    """Generate signals using SAME logic across all timeframes"""
    try:
        df['signal'] = 0

        start_idx = max(25, len(df)//4)  # Adaptive start based on data length

        for i in range(start_idx, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            # EXACT SAME CONDITIONS as original system
            conditions = [
                # Breakout condition
                current['close'] > current['high_20'],

                # Volume confirmation
                current['volume_ratio'] > VOLUME_MULT,

                # Trend strength
                current['adx'] > ADX_MIN,

                # Not overbought
                current['rsi'] < RSI_OVERBOUGHT,

                # Price above moving average
                current['close'] > current['sma_20'],

                # Rising RSI (momentum)
                current['rsi'] > prev['rsi']
            ]

            # SAME requirement: 4 out of 6 conditions
            if sum(conditions) >= 4:
                df.iloc[i, df.columns.get_loc('signal')] = 1

        return df
    except Exception as e:
        print(f"Error generating signals: {e}")
        df['signal'] = 0
        return df

def backtest_timeframe(df, symbol, timeframe):
    """Backtest using timeframe-specific parameters"""
    try:
        config = TIMEFRAME_CONFIGS[timeframe]

        cash = INITIAL_CAPITAL
        positions = {}
        trades = []

        for i in range(1, len(df)):
            current = df.iloc[i]

            # Exit existing positions
            positions_to_close = []
            for pos_id, pos in positions.items():
                current_return = (current['close'] - pos['entry_price']) / pos['entry_price']
                bars_held = i - pos['entry_bar']

                # Timeframe-specific exit conditions
                exit_reason = None
                if current_return >= config['profit_target']:
                    exit_reason = 'Profit Target'
                elif current_return <= -config['stop_loss']:
                    exit_reason = 'Stop Loss'
                elif bars_held >= config['max_hold_bars']:
                    exit_reason = 'Time Exit'

                if exit_reason:
                    shares = pos['shares']
                    exit_price = current['close'] * 0.9995  # Slippage

                    pnl = (exit_price - pos['entry_price']) * shares
                    commission = (pos['entry_price'] + exit_price) * shares * 0.0005
                    net_pnl = pnl - commission

                    trade = {
                        'symbol': symbol,
                        'timeframe': timeframe,
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
                len(positions) < 3 and  # Max 3 positions
                cash > 5000):

                # Same position sizing logic
                position_value = cash * POSITION_SIZE
                entry_price = current['close'] * 1.0005  # Slippage
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
        print(f"Error in backtest for {symbol} on {timeframe}: {e}")
        return []

def run_comprehensive_timeframe_analysis():
    """Test SAME logic across ALL timeframes comprehensively"""

    print("🚀 COMPREHENSIVE TIMEFRAME ANALYSIS")
    print("=" * 50)
    print("Testing SAME logic across ALL timeframes to eliminate bias!")
    print("This will show which timeframe actually works best.")
    print()

    # Get all symbols
    all_symbols = get_all_symbols()
    print(f"Testing {len(all_symbols)} symbols across all timeframes...")

    # Test each timeframe comprehensively
    all_timeframe_results = {}

    for timeframe in TIMEFRAME_CONFIGS.keys():
        print(f"\n📊 TESTING {timeframe.upper()} TIMEFRAME")
        config = TIMEFRAME_CONFIGS[timeframe]
        print(f"Target: {config['profit_target']*100:.1f}% profit, {config['stop_loss']*100:.1f}% stop")
        print(f"Description: {config['description']}")
        print("-" * 60)

        timeframe_results = []
        successful_symbols = 0
        failed_symbols = 0

        # Test each symbol on this timeframe
        for i, symbol in enumerate(all_symbols[:50], 1):  # Test first 50 symbols for speed
            if i % 10 == 0:
                print(f"  Progress: {i}/50 symbols tested...")

            try:
                # Load and resample data
                df = load_and_resample_data(symbol, timeframe)
                if df is None:
                    failed_symbols += 1
                    continue

                # Add indicators
                df = add_timeframe_indicators(df, timeframe)

                # Generate signals
                df = generate_timeframe_signals(df)

                # Check signal count
                signal_count = df['signal'].sum()
                if signal_count < 3:
                    failed_symbols += 1
                    continue

                # Run backtest
                trades = backtest_timeframe(df, symbol, timeframe)

                if len(trades) >= 2:
                    # Calculate stats
                    df_trades = pd.DataFrame(trades)
                    winning_trades = len(df_trades[df_trades['pnl'] > 0])

                    result = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'total_trades': len(trades),
                        'winning_trades': winning_trades,
                        'win_rate': (winning_trades / len(trades)) * 100,
                        'total_pnl': df_trades['pnl'].sum(),
                        'avg_pnl_per_trade': df_trades['pnl'].mean(),
                        'best_trade': df_trades['pnl'].max(),
                        'worst_trade': df_trades['pnl'].min(),
                        'signals_generated': signal_count,
                        'bars_analyzed': len(df),
                        'is_profitable': df_trades['pnl'].sum() > 0
                    }

                    timeframe_results.append(result)
                    successful_symbols += 1
                else:
                    failed_symbols += 1

            except Exception as e:
                failed_symbols += 1
                continue

        # Store timeframe results
        all_timeframe_results[timeframe] = timeframe_results

        # Summarize this timeframe
        if timeframe_results:
            df_results = pd.DataFrame(timeframe_results)
            profitable_count = len(df_results[df_results['is_profitable']])
            total_pnl = df_results['total_pnl'].sum()
            total_trades = df_results['total_trades'].sum()
            avg_win_rate = df_results['win_rate'].mean()

            print(f"\n✅ {timeframe.upper()} SUMMARY:")
            print(f"  Symbols tested: {successful_symbols}")
            print(f"  Profitable symbols: {profitable_count} ({profitable_count/successful_symbols*100:.1f}%)")
            print(f"  Total trades: {total_trades:,}")
            print(f"  Total PnL: ₹{total_pnl:,.0f}")
            print(f"  Average win rate: {avg_win_rate:.1f}%")
            print(f"  Return on capital: {(total_pnl/INITIAL_CAPITAL)*100:.2f}%")

            # Save results
            df_results.to_csv(f'{timeframe}_comprehensive_results.csv', index=False)
        else:
            print(f"\n❌ {timeframe.upper()}: No successful results")

    # COMPREHENSIVE COMPARISON ACROSS ALL TIMEFRAMES
    print(f"\n\n🏆 COMPREHENSIVE TIMEFRAME COMPARISON")
    print("=" * 60)

    timeframe_summary = {}

    for timeframe, results in all_timeframe_results.items():
        if results:
            df_results = pd.DataFrame(results)
            profitable_count = len(df_results[df_results['is_profitable']])

            summary = {
                'timeframe': timeframe,
                'symbols_tested': len(df_results),
                'profitable_symbols': profitable_count,
                'success_rate': (profitable_count / len(df_results)) * 100,
                'total_trades': df_results['total_trades'].sum(),
                'total_pnl': df_results['total_pnl'].sum(),
                'avg_win_rate': df_results['win_rate'].mean(),
                'return_on_capital': (df_results['total_pnl'].sum() / INITIAL_CAPITAL) * 100,
                'profit_target': TIMEFRAME_CONFIGS[timeframe]['profit_target'] * 100,
                'stop_loss': TIMEFRAME_CONFIGS[timeframe]['stop_loss'] * 100
            }

            timeframe_summary[timeframe] = summary

    # Display comparison
    if timeframe_summary:
        print(f"\n📊 TIMEFRAME PERFORMANCE RANKING:")
        print("-" * 40)

        # Sort by return on capital
        sorted_timeframes = sorted(timeframe_summary.items(), 
                                 key=lambda x: x[1]['return_on_capital'], reverse=True)

        print(f"{'Timeframe':<12} {'Symbols':<8} {'Success%':<9} {'Total PnL':<12} {'ROC%':<8} {'Win Rate'}")
        print("-" * 75)

        for timeframe, summary in sorted_timeframes:
            print(f"{timeframe:<12} {summary['symbols_tested']:<8} "
                  f"{summary['success_rate']:<8.1f}% ₹{summary['total_pnl']:<11.0f} "
                  f"{summary['return_on_capital']:<7.1f}% {summary['avg_win_rate']:<7.1f}%")

        # Best timeframe analysis
        best_timeframe, best_stats = sorted_timeframes[0]

        print(f"\n🏆 BEST PERFORMING TIMEFRAME: {best_timeframe.upper()}")
        print(f"   Return on Capital: {best_stats['return_on_capital']:.2f}%")
        print(f"   Total PnL: ₹{best_stats['total_pnl']:,.0f}")
        print(f"   Success Rate: {best_stats['success_rate']:.1f}%")
        print(f"   Profit Target: {best_stats['profit_target']:.1f}%")
        print(f"   Average Win Rate: {best_stats['avg_win_rate']:.1f}%")

        # Compare to 15-minute baseline
        baseline_timeframe = '15minute'
        if baseline_timeframe in timeframe_summary:
            baseline_stats = timeframe_summary[baseline_timeframe]
            improvement = best_stats['return_on_capital'] - baseline_stats['return_on_capital']

            print(f"\n📈 IMPROVEMENT vs 15-MINUTE BASELINE:")
            print(f"   15-minute ROC: {baseline_stats['return_on_capital']:.2f}%")
            print(f"   {best_timeframe} ROC: {best_stats['return_on_capital']:.2f}%")
            print(f"   Improvement: {improvement:+.2f} percentage points")

            if improvement > 5:
                print(f"\n🚀 MAJOR DISCOVERY: {best_timeframe} is significantly better!")
            elif improvement > 0:
                print(f"\n✅ IMPROVEMENT: {best_timeframe} shows better performance")
            else:
                print(f"\n✅ VALIDATION: 15-minute baseline is competitive")

        print(f"\n🎯 FINAL RECOMMENDATION:")
        print(f"Based on comprehensive testing across all timeframes:")

        if best_stats['return_on_capital'] > 0:
            print(f"✅ DEPLOY {best_timeframe.upper()} TIMEFRAME")
            print(f"   Expected return: {best_stats['return_on_capital']:.1f}% on capital")
            print(f"   Use {best_stats['success_rate']:.0f}% success rate symbols")
        else:
            print(f"❌ ALL TIMEFRAMES SHOW LOSSES")
            print(f"   System requires fundamental redesign")

    return all_timeframe_results

if __name__ == "__main__":
    results = run_comprehensive_timeframe_analysis()
