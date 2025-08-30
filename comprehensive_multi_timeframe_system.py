
import os
import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# COMPREHENSIVE MULTI-TIMEFRAME TRADING SYSTEM
# =============================================================================

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data")
}

# CORE PARAMETERS (SAME ACROSS ALL TIMEFRAMES)
INITIAL_CAPITAL = 100000
POSITION_SIZE_PER_TIMEFRAME = 0.02  # 2% per timeframe (can run 5 timeframes = 10% total)
YEAR_FILTER = 2025

# TECHNICAL PARAMETERS (CONSISTENT LOGIC)
RSI_OVERBOUGHT = 70
ADX_MIN = 20
VOLUME_MULT = 1.5

# COMPREHENSIVE TIMEFRAME CONFIGURATIONS
TIMEFRAME_CONFIGS = {
    '15minute': {
        'profit_target': 0.025,   # 2.5% (verified working)
        'stop_loss': 0.015,       # 1.5%
        'max_hold_bars': 25,      # ~6 hours
        'data_source': '15minute',
        'description': 'Scalping trades',
        'expected_frequency': 'High (2-3 trades/day)',
        'capital_allocation': 0.02  # 2% of total capital
    },
    '1hour': {
        'profit_target': 0.035,   # 3.5% (larger intraday moves)
        'stop_loss': 0.020,       # 2.0%
        'max_hold_bars': 12,      # ~12 hours (1.5 days)
        'data_source': '1hour',
        'description': 'Short swing trades',
        'expected_frequency': 'Medium (1 trade/day)',
        'capital_allocation': 0.02  # 2% of total capital
    },
    'daily': {
        'profit_target': 0.060,   # 6.0% (daily swing moves)
        'stop_loss': 0.030,       # 3.0%
        'max_hold_bars': 5,       # ~5 days
        'data_source': 'daily',
        'description': 'Medium swing trades',
        'expected_frequency': 'Medium (3-4 trades/week)',
        'capital_allocation': 0.02  # 2% of total capital
    },
    'weekly': {
        'profit_target': 0.100,   # 10.0% (weekly trend moves)
        'stop_loss': 0.050,       # 5.0%
        'max_hold_bars': 4,       # ~4 weeks
        'data_source': 'daily',   # Weekly derived from daily
        'description': 'Long swing trades',
        'expected_frequency': 'Low (1-2 trades/month)',
        'capital_allocation': 0.03  # 3% of total capital
    },
    'monthly': {
        'profit_target': 0.150,   # 15.0% (monthly trend moves)
        'stop_loss': 0.075,       # 7.5%
        'max_hold_bars': 3,       # ~3 months
        'data_source': 'daily',   # Monthly derived from daily
        'description': 'Position trades',
        'expected_frequency': 'Very Low (1-2 trades/quarter)',
        'capital_allocation': 0.03  # 3% of total capital
    }
}

PROFITABLE_SYMBOLS = ['FINPIPE', 'GREENLAM', 'WABAG', 'ITI', 'LTTS', 'CONCORDBIO']

def resample_to_timeframe(df, target_timeframe):
    """Convert daily data to weekly/monthly timeframes"""
    try:
        if target_timeframe == 'weekly':
            # Resample to weekly (Monday start)
            df_resampled = df.set_index('date').resample('W-MON').agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna().reset_index()

        elif target_timeframe == 'monthly':
            # Resample to monthly (month start)
            df_resampled = df.set_index('date').resample('MS').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min', 
                'close': 'last',
                'volume': 'sum'
            }).dropna().reset_index()

        else:
            return df

        return df_resampled

    except Exception as e:
        print(f"Error resampling to {target_timeframe}: {e}")
        return df

def load_timeframe_data(symbol, timeframe):
    """Load and prepare data for specific timeframe"""
    try:
        config = TIMEFRAME_CONFIGS[timeframe]
        data_source = config['data_source']

        # Load base data
        df = pd.read_csv(os.path.join(DATA_PATHS[data_source], f"{symbol}.csv"))
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)

        # Resample if needed for weekly/monthly
        if timeframe in ['weekly', 'monthly']:
            df = resample_to_timeframe(df, timeframe)

        # Minimum data requirement varies by timeframe
        min_bars = {'15minute': 100, '1hour': 80, 'daily': 60, 'weekly': 40, 'monthly': 20}

        if len(df) < min_bars.get(timeframe, 50):
            return None

        return df
    except Exception as e:
        print(f"Error loading {timeframe} data for {symbol}: {e}")
        return None

def add_comprehensive_indicators(df, timeframe):
    """Add indicators optimized for each timeframe"""
    try:
        # Timeframe-adjusted periods
        if timeframe == '15minute':
            periods = {'rsi': 14, 'adx': 14, 'sma': 20, 'high': 20}
        elif timeframe == '1hour':
            periods = {'rsi': 14, 'adx': 14, 'sma': 20, 'high': 20}
        elif timeframe == 'daily':
            periods = {'rsi': 14, 'adx': 14, 'sma': 20, 'high': 20}
        elif timeframe == 'weekly':
            periods = {'rsi': 10, 'adx': 10, 'sma': 10, 'high': 10}  # Shorter for weekly
        else:  # monthly
            periods = {'rsi': 6, 'adx': 6, 'sma': 6, 'high': 6}    # Shorter for monthly

        # Calculate indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=periods['rsi'])
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=periods['adx'])

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(periods['sma']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price indicators
        df['sma_20'] = df['close'].rolling(periods['sma']).mean()
        df['high_20'] = df['high'].rolling(periods['high']).max()

        # Additional trend indicators for longer timeframes
        if timeframe in ['weekly', 'monthly']:
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=min(periods['sma']*2, len(df)//3))
            df['trend_strength'] = (df['close'] - df['ema_50']) / df['ema_50']

        # Fill NaN values
        numeric_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        for col in numeric_columns:
            if col in ['rsi', 'adx', 'volume_ratio', 'sma_20', 'high_20']:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        return df
    except Exception as e:
        print(f"Error adding indicators for {timeframe}: {e}")
        return df

def generate_comprehensive_signals(df, timeframe):
    """Generate signals with timeframe-appropriate logic"""
    try:
        df['signal'] = 0

        start_idx = {'15minute': 50, '1hour': 40, 'daily': 30, 'weekly': 20, 'monthly': 10}

        for i in range(start_idx.get(timeframe, 30), len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            # Core signal conditions (same logic, different sensitivity)
            conditions = [
                # Breakout condition
                current['close'] > current['high_20'],

                # Volume confirmation (adjusted for timeframe)
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

            # Additional conditions for longer timeframes
            if timeframe in ['weekly', 'monthly'] and 'trend_strength' in df.columns:
                conditions.append(current['trend_strength'] > 0.02)  # Strong trend
                required_conditions = 5  # 5 out of 7 for longer timeframes
            else:
                required_conditions = 4  # 4 out of 6 for shorter timeframes

            # Generate signal
            if sum(conditions) >= required_conditions:
                df.iloc[i, df.columns.get_loc('signal')] = 1

        return df
    except Exception as e:
        print(f"Error generating signals for {timeframe}: {e}")
        df['signal'] = 0
        return df

def backtest_comprehensive_timeframe(df, symbol, timeframe):
    """Backtest with comprehensive timeframe parameters"""
    try:
        config = TIMEFRAME_CONFIGS[timeframe]
        capital = INITIAL_CAPITAL * config['capital_allocation']  # Allocated capital

        cash = capital
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
                        'bars_held': bars_held,
                        'capital_allocated': capital
                    }

                    trades.append(trade)
                    cash += pos['entry_value'] + net_pnl
                    positions_to_close.append(pos_id)

            # Remove closed positions
            for pos_id in positions_to_close:
                del positions[pos_id]

            # Check for new entries (only one position per timeframe per symbol)
            if (current['signal'] == 1 and
                len(positions) == 0 and  # One position per symbol per timeframe
                cash > 1000):

                # Use available capital for position
                position_value = min(cash * 0.8, cash - 1000)  # Keep some cash buffer
                entry_price = current['close'] * 1.0005  # Slippage
                shares = position_value / entry_price

                if shares > 0:
                    positions[0] = {
                        'entry_date': current['date'],
                        'entry_price': entry_price,
                        'shares': shares,
                        'entry_bar': i,
                        'entry_value': position_value
                    }
                    cash -= position_value

        return trades

    except Exception as e:
        print(f"Error in {timeframe} backtest for {symbol}: {e}")
        return []

def run_comprehensive_multi_timeframe_test():
    """Test all timeframes comprehensively"""

    print("🚀 COMPREHENSIVE MULTI-TIMEFRAME SYSTEM")
    print("=" * 60)
    print("Testing 15min → 1hour → Daily → Weekly → Monthly timeframes!")
    print()

    # Display timeframe configurations
    print("⚙️ TIMEFRAME CONFIGURATIONS:")
    print("-" * 30)
    print(f"{'Timeframe':<12} {'Profit Target':<13} {'Stop Loss':<10} {'Max Hold':<9} {'Capital %'}")
    print("-" * 70)

    for tf, config in TIMEFRAME_CONFIGS.items():
        print(f"{tf:<12} {config['profit_target']*100:<13.1f}% "
              f"{config['stop_loss']*100:<10.1f}% "
              f"{config['max_hold_bars']:<9} "
              f"{config['capital_allocation']*100:<8.0f}%")

    all_results = {}
    timeframe_summaries = {}

    # Test each timeframe
    for timeframe in ['15minute', '1hour', 'daily', 'weekly', 'monthly']:
        print(f"\n📊 TESTING {timeframe.upper()} TIMEFRAME:")
        print(f"Target: {TIMEFRAME_CONFIGS[timeframe]['profit_target']*100:.1f}% profit, "
              f"{TIMEFRAME_CONFIGS[timeframe]['stop_loss']*100:.1f}% stop")
        print("-" * 60)

        timeframe_results = []

        for i, symbol in enumerate(PROFITABLE_SYMBOLS, 1):
            print(f"[{i}/{len(PROFITABLE_SYMBOLS)}] Processing {symbol}...")

            try:
                # Load timeframe data
                df = load_timeframe_data(symbol, timeframe)
                if df is None:
                    print(f"  ✗ Insufficient {timeframe} data")
                    continue

                # Add indicators
                df = add_comprehensive_indicators(df, timeframe)

                # Generate signals
                df = generate_comprehensive_signals(df, timeframe)

                # Check signal count
                signal_count = df['signal'].sum()
                if signal_count < 1:
                    print(f"  ✗ No signals generated")
                    continue

                print(f"  📊 {signal_count} signals, {len(df)} {timeframe} bars")

                # Run backtest
                trades = backtest_comprehensive_timeframe(df, symbol, timeframe)

                if len(trades) >= 1:
                    # Calculate stats
                    df_trades = pd.DataFrame(trades)
                    winning_trades = len(df_trades[df_trades['pnl'] > 0])

                    result = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'total_trades': len(trades),
                        'winning_trades': winning_trades,
                        'win_rate': (winning_trades / len(trades)) * 100 if len(trades) > 0 else 0,
                        'total_pnl': df_trades['pnl'].sum(),
                        'avg_pnl_per_trade': df_trades['pnl'].mean(),
                        'best_trade': df_trades['pnl'].max(),
                        'worst_trade': df_trades['pnl'].min(),
                        'avg_return_pct': df_trades['return_pct'].mean() * 100,
                        'avg_hold_bars': df_trades['bars_held'].mean(),
                        'signals_generated': signal_count,
                        'bars_analyzed': len(df),
                        'capital_allocated': df_trades['capital_allocated'].iloc[0]
                    }

                    timeframe_results.append(result)

                    print(f"  ✅ {result['total_trades']} trades, "
                          f"{result['win_rate']:.1f}% win rate, "
                          f"₹{result['total_pnl']:.0f} PnL")
                else:
                    print(f"  ✗ No valid trades generated")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        # Store and summarize timeframe results
        if timeframe_results:
            all_results[timeframe] = timeframe_results
            df_results = pd.DataFrame(timeframe_results)

            # Timeframe summary
            summary = {
                'timeframe': timeframe,
                'successful_symbols': len(df_results),
                'total_trades': df_results['total_trades'].sum(),
                'total_pnl': df_results['total_pnl'].sum(),
                'avg_win_rate': df_results['win_rate'].mean(),
                'avg_pnl_per_trade': df_results['avg_pnl_per_trade'].mean(),
                'capital_allocated': TIMEFRAME_CONFIGS[timeframe]['capital_allocation'] * INITIAL_CAPITAL,
                'return_on_allocated_capital': (df_results['total_pnl'].sum() / (TIMEFRAME_CONFIGS[timeframe]['capital_allocation'] * INITIAL_CAPITAL)) * 100
            }

            timeframe_summaries[timeframe] = summary

            # Save results
            df_results.to_csv(f'{timeframe}_comprehensive_results.csv', index=False)
            print(f"\n✅ {timeframe} SUMMARY: {summary['total_trades']} trades, "
                  f"₹{summary['total_pnl']:.0f} profit, "
                  f"{summary['return_on_allocated_capital']:.1f}% return on allocated capital")
        else:
            print(f"\n❌ No successful results for {timeframe}")

    # COMPREHENSIVE ANALYSIS
    print(f"\n\n🏆 COMPREHENSIVE MULTI-TIMEFRAME ANALYSIS:")
    print("=" * 65)

    if timeframe_summaries:
        print(f"\n📊 PERFORMANCE BY TIMEFRAME:")
        print("-" * 35)
        print(f"{'Timeframe':<12} {'Symbols':<8} {'Trades':<7} {'Total PnL':<11} {'Allocated':<11} {'ROI %'}")
        print("-" * 80)

        total_system_pnl = 0
        total_allocated_capital = 0

        for tf, summary in timeframe_summaries.items():
            total_system_pnl += summary['total_pnl']
            total_allocated_capital += summary['capital_allocated']

            print(f"{tf:<12} {summary['successful_symbols']:<8} {summary['total_trades']:<7} "
                  f"₹{summary['total_pnl']:<10.0f} ₹{summary['capital_allocated']:<10.0f} "
                  f"{summary['return_on_allocated_capital']:<6.1f}%")

        # System-wide analysis
        print(f"\n🎯 SYSTEM-WIDE PERFORMANCE:")
        print("-" * 28)
        print(f"Total System PnL: ₹{total_system_pnl:.0f}")
        print(f"Total Capital Used: ₹{total_allocated_capital:.0f} ({total_allocated_capital/INITIAL_CAPITAL*100:.0f}% of ₹{INITIAL_CAPITAL:,})")
        print(f"Overall System ROI: {(total_system_pnl/total_allocated_capital)*100:.1f}%")

        # Monthly projections
        print(f"\n💰 MONTHLY RETURN PROJECTIONS:")
        print("-" * 30)

        monthly_projections = {}
        total_monthly_pnl = 0

        for tf, summary in timeframe_summaries.items():
            if tf == '15minute':
                monthly_factor = 22 / 250  # 22 trading days per month
            elif tf == '1hour':
                monthly_factor = 22 / 250
            elif tf == 'daily':
                monthly_factor = 22 / 250
            elif tf == 'weekly':
                monthly_factor = 4.33 / 52  # ~4.33 weeks per month
            else:  # monthly
                monthly_factor = 1 / 12  # 1 month per month

            monthly_pnl = summary['total_pnl'] * monthly_factor
            monthly_return_pct = (monthly_pnl / summary['capital_allocated']) * 100

            monthly_projections[tf] = {
                'monthly_pnl': monthly_pnl,
                'monthly_return_pct': monthly_return_pct
            }

            total_monthly_pnl += monthly_pnl

            print(f"{tf:<12}: ₹{monthly_pnl:>6.0f}/month ({monthly_return_pct:>5.1f}% on allocated capital)")

        total_monthly_return_pct = (total_monthly_pnl / total_allocated_capital) * 100

        print(f"\n🚀 COMBINED SYSTEM MONTHLY POTENTIAL:")
        print(f"Total Monthly PnL: ₹{total_monthly_pnl:.0f}")
        print(f"Monthly Return: {total_monthly_return_pct:.1f}% on ₹{total_allocated_capital:,.0f} allocated")
        print(f"Annualized Return: {total_monthly_return_pct * 12:.1f}%")

        # Best timeframe recommendation
        best_timeframe = max(timeframe_summaries.items(), key=lambda x: x[1]['return_on_allocated_capital'])
        print(f"\n⭐ BEST PERFORMING TIMEFRAME: {best_timeframe[0].upper()}")
        print(f"   ROI: {best_timeframe[1]['return_on_allocated_capital']:.1f}%")
        print(f"   Total Profit: ₹{best_timeframe[1]['total_pnl']:.0f}")
        print(f"   Trade Count: {best_timeframe[1]['total_trades']}")

        print(f"\n🎯 IMPLEMENTATION RECOMMENDATION:")
        print("-" * 32)
        if total_monthly_return_pct > 10:  # If system shows >10% monthly potential
            print("🚀 DEPLOY MULTI-TIMEFRAME SYSTEM!")
            print(f"   Potential: {total_monthly_return_pct:.1f}% monthly return")
            print(f"   This is {total_monthly_return_pct/3:.1f}x better than 15-min only!")
            print("   Implement all profitable timeframes in parallel")
        else:
            print("✅ Stick with 15-minute verified system")
            print("   Multi-timeframe doesn't show significant improvement")
    else:
        print("❌ No successful results across any timeframe")

    return all_results, timeframe_summaries

if __name__ == "__main__":
    results, summaries = run_comprehensive_multi_timeframe_test()
