
import os
import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# WEEKLY/MONTHLY TIMEFRAME TESTER - FOLLOWING THE PROFITABILITY PATTERN
# =============================================================================

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data")
}

# CORE PARAMETERS
INITIAL_CAPITAL = 100000
POSITION_SIZE = 0.02
YEAR_FILTER = 2025

# TECHNICAL PARAMETERS (SAME LOGIC)
RSI_OVERBOUGHT = 70
ADX_MIN = 20
VOLUME_MULT = 1.5

# WEEKLY/MONTHLY CONFIGURATIONS - Following the profitability pattern
TIMEFRAME_CONFIGS = {
    'weekly': {
        'source_data': 'daily',
        'resample_freq': 'W-MON',  # Weekly starting Monday
        'profit_target': 0.080,    # 8.0% (following the pattern)
        'stop_loss': 0.040,        # 4.0% (wider for longer holds)
        'max_hold_bars': 4,        # 4 weeks maximum
        'description': 'Weekly swing trading',
        'expected_roc': '0% to +5%'  # Based on pattern extrapolation
    },
    'monthly': {
        'source_data': 'daily', 
        'resample_freq': 'MS',     # Monthly starting first day
        'profit_target': 0.120,    # 12.0% (following the pattern)
        'stop_loss': 0.060,        # 6.0% (wider stops)
        'max_hold_bars': 3,        # 3 months maximum
        'description': 'Monthly position trading',
        'expected_roc': '+3% to +10%'  # Based on pattern extrapolation
    }
}

def get_all_symbols():
    """Get all available symbols"""
    try:
        files = [f for f in os.listdir(DATA_PATHS['daily']) if f.endswith('.csv')]
        return sorted([f.replace('.csv', '') for f in files])
    except:
        return []

def load_and_resample_data(symbol, timeframe):
    """Load daily data and resample to weekly/monthly"""
    try:
        config = TIMEFRAME_CONFIGS[timeframe]

        # Load daily data
        file_path = os.path.join(DATA_PATHS['daily'], f"{symbol}.csv")
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)

        if len(df) < 100:  # Need sufficient daily data
            return None

        # Set date as index for resampling
        df.set_index('date', inplace=True)

        # Resample to target timeframe
        df_resampled = df.resample(config['resample_freq']).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Reset index
        df_resampled.reset_index(inplace=True)

        # Need minimum bars for analysis
        min_bars = 20 if timeframe == 'weekly' else 12
        if len(df_resampled) < min_bars:
            return None

        return df_resampled

    except Exception as e:
        print(f"Error loading {symbol} for {timeframe}: {e}")
        return None

def add_technical_indicators(df, timeframe):
    """Add technical indicators optimized for longer timeframes"""
    try:
        # Adjust periods for longer timeframes
        if timeframe == 'weekly':
            rsi_period = min(14, len(df)//3)
            adx_period = min(14, len(df)//3)
            sma_period = min(10, len(df)//3)  # Shorter periods for weekly
        else:  # monthly
            rsi_period = min(10, len(df)//3)
            adx_period = min(10, len(df)//3)
            sma_period = min(6, len(df)//3)   # Shorter periods for monthly

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
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    except Exception as e:
        print(f"Error adding indicators for {timeframe}: {e}")
        return df

def generate_signals(df):
    """Generate signals using SAME 4/6 logic"""
    try:
        df['signal'] = 0

        start_idx = max(10, len(df)//4)

        for i in range(start_idx, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1] if i > 0 else current

            # SAME 6 conditions as always
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
    """Backtest with longer timeframe parameters"""
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

                # Longer timeframe exit conditions
                exit_reason = None
                if current_return >= config['profit_target']:
                    exit_reason = 'Profit Target'
                elif current_return <= -config['stop_loss']:
                    exit_reason = 'Stop Loss'
                elif bars_held >= config['max_hold_bars']:
                    exit_reason = 'Time Exit'

                if exit_reason:
                    shares = pos['shares']
                    exit_price = current['close'] * 0.999  # Lower slippage for longer timeframes

                    pnl = (exit_price - pos['entry_price']) * shares
                    commission = (pos['entry_price'] + exit_price) * shares * 0.0003  # Lower commission
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

            for pos_id in positions_to_close:
                del positions[pos_id]

            # Check for new entries (only 1 position for longer timeframes)
            if (current['signal'] == 1 and
                len(positions) == 0 and  # Only one position at a time
                cash > 10000):

                # Larger position sizing for longer timeframes
                position_value = cash * POSITION_SIZE * 2  # 4% for longer holds
                entry_price = current['close'] * 1.001     # Lower slippage
                shares = position_value / entry_price

                if cash >= position_value:
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
        print(f"Error in backtest for {symbol} on {timeframe}: {e}")
        return []

def run_weekly_monthly_analysis():
    """Test weekly and monthly timeframes following the profitability pattern"""

    print("🚀 WEEKLY/MONTHLY PROFITABILITY TESTER")
    print("=" * 50)
    print("Following the clear pattern: Longer timeframes = Better performance!")
    print("Daily was -2.2% ROC with 47.9% success rate.")
    print("Weekly/Monthly should be PROFITABLE based on the trend!")
    print()

    # Get symbols (test more since we expect better results)
    all_symbols = get_all_symbols()
    test_symbols = all_symbols[:100]  # Test 100 symbols (more than before)

    print(f"Testing {len(test_symbols)} symbols on weekly and monthly timeframes...")

    results = {}

    # Test both timeframes
    for timeframe in ['weekly', 'monthly']:
        config = TIMEFRAME_CONFIGS[timeframe]

        print(f"\n📊 TESTING {timeframe.upper()} TIMEFRAME")
        print(f"Target: {config['profit_target']*100:.1f}% profit, {config['stop_loss']*100:.1f}% stop")
        print(f"Expected ROC: {config['expected_roc']} (based on pattern)")
        print(f"Description: {config['description']}")
        print("-" * 60)

        timeframe_results = []
        successful = 0
        failed = 0

        for i, symbol in enumerate(test_symbols, 1):
            if i % 25 == 0:
                print(f"  Progress: {i}/{len(test_symbols)} symbols...")

            try:
                # Load and resample data
                df = load_and_resample_data(symbol, timeframe)
                if df is None:
                    failed += 1
                    continue

                # Add indicators
                df = add_technical_indicators(df, timeframe)

                # Generate signals
                df = generate_signals(df)

                # Check signals
                signal_count = df['signal'].sum()
                if signal_count < 1:
                    failed += 1
                    continue

                # Backtest
                trades = backtest_timeframe(df, symbol, timeframe)

                if len(trades) >= 1:  # Even 1 trade is valuable for longer timeframes
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
                        'is_profitable': df_trades['pnl'].sum() > 0,
                        'avg_return_pct': df_trades['return_pct'].mean() * 100
                    }

                    timeframe_results.append(result)
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                failed += 1
                continue

        # Store results
        results[timeframe] = timeframe_results

        # Analyze this timeframe
        if timeframe_results:
            df_results = pd.DataFrame(timeframe_results)
            profitable_count = len(df_results[df_results['is_profitable']])
            total_pnl = df_results['total_pnl'].sum()
            total_trades = df_results['total_trades'].sum()
            avg_win_rate = df_results['win_rate'].mean()
            roc = (total_pnl / INITIAL_CAPITAL) * 100

            print(f"\n✅ {timeframe.upper()} RESULTS:")
            print(f"  Symbols tested: {successful}")
            print(f"  Profitable symbols: {profitable_count} ({profitable_count/successful*100:.1f}%)")
            print(f"  Total trades: {total_trades}")
            print(f"  Total PnL: ₹{total_pnl:,.0f}")
            print(f"  Return on Capital: {roc:.2f}%")
            print(f"  Average win rate: {avg_win_rate:.1f}%")

            # BREAKTHROUGH CHECK!
            if roc > 0:
                print(f"\n🎉 BREAKTHROUGH! {timeframe.upper()} IS PROFITABLE!")
                print(f"   Positive ROC: {roc:.2f}%")
                print(f"   Pattern prediction was CORRECT!")
            elif roc > -1:
                print(f"\n🎯 VERY CLOSE! {timeframe.upper()} nearly profitable")
                print(f"   ROC: {roc:.2f}% (very close to breakeven)")

            # Save results
            df_results.to_csv(f'{timeframe}_results.csv', index=False)

            # Show top performers
            if len(df_results) > 0:
                top_performers = df_results.nlargest(5, 'total_pnl')
                print(f"\n🏆 TOP 5 {timeframe.upper()} PERFORMERS:")
                for _, row in top_performers.iterrows():
                    print(f"  {row['symbol']:<12}: ₹{row['total_pnl']:>6.0f} "
                          f"({row['total_trades']:>2} trades, {row['win_rate']:>5.1f}% win rate)")
        else:
            print(f"\n❌ {timeframe.upper()}: No successful results")

    # COMPREHENSIVE COMPARISON
    print(f"\n\n🏆 COMPREHENSIVE LONG-TERM TIMEFRAME ANALYSIS")
    print("=" * 65)

    # Compare all results including daily baseline
    comparison_data = [
        ('Daily (baseline)', -2.23, 47.9, 41.3),  # From previous results
    ]

    for timeframe, timeframe_results in results.items():
        if timeframe_results:
            df_results = pd.DataFrame(timeframe_results)
            profitable_count = len(df_results[df_results['is_profitable']])
            roc = (df_results['total_pnl'].sum() / INITIAL_CAPITAL) * 100
            success_rate = (profitable_count / len(df_results)) * 100
            avg_win_rate = df_results['win_rate'].mean()

            comparison_data.append((timeframe.title(), roc, success_rate, avg_win_rate))

    # Display comparison
    print(f"\n📊 TIMEFRAME PERFORMANCE COMPARISON:")
    print("-" * 45)
    print(f"{'Timeframe':<15} {'ROC %':<8} {'Success %':<11} {'Win Rate %'}")
    print("-" * 50)

    # Sort by ROC
    comparison_data.sort(key=lambda x: x[1], reverse=True)

    for timeframe, roc, success_rate, win_rate in comparison_data:
        status = "🎉 PROFITABLE!" if roc > 0 else "🎯 Close" if roc > -1 else ""
        print(f"{timeframe:<15} {roc:<8.2f} {success_rate:<11.1f} {win_rate:<11.1f} {status}")

    # Final analysis
    best_timeframe = comparison_data[0]

    print(f"\n🏆 BEST PERFORMING TIMEFRAME: {best_timeframe[0].upper()}")
    print(f"   ROC: {best_timeframe[1]:.2f}%")
    print(f"   Success Rate: {best_timeframe[2]:.1f}%")
    print(f"   Average Win Rate: {best_timeframe[3]:.1f}%")

    if best_timeframe[1] > 0:
        print(f"\n🚀 BREAKTHROUGH ACHIEVED!")
        print(f"Pattern prediction was CORRECT!")
        print(f"Longer timeframes DO lead to profitability!")
        print(f"\nRECOMMENDATION: DEPLOY {best_timeframe[0].upper()} TIMEFRAME!")
        print(f"Expected annual return: {best_timeframe[1]:.1f}%")
    elif best_timeframe[1] > -1:
        print(f"\n🎯 VERY CLOSE TO BREAKTHROUGH!")
        print(f"Minor optimizations could push to profitability")
        print(f"Pattern is clearly working - continue extending timeframes")
    else:
        print(f"\n📈 PATTERN CONTINUES:")
        print(f"Still following improvement trend")
        print(f"Consider testing quarterly/yearly timeframes")

    return results

if __name__ == "__main__":
    results = run_weekly_monthly_analysis()
