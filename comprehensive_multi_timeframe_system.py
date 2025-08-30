
import os
import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# COMPREHENSIVE ALL-SYMBOLS TESTING SYSTEM
# =============================================================================

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data")
}

# ORIGINAL VERIFIED PARAMETERS
INITIAL_CAPITAL = 100000
MAX_POSITIONS = 3
POSITION_SIZE = 0.02
YEAR_FILTER = 2025

# Technical parameters
RSI_OVERBOUGHT = 70
ADX_MIN = 20
VOLUME_MULT = 1.5
PROFIT_TARGET = 0.025  # 2.5%
STOP_LOSS = 0.015      # 1.5%
MAX_HOLD_BARS = 25

def get_all_available_symbols():
    """Get ALL symbols from data directories"""
    all_symbols = set()

    for data_type, path in DATA_PATHS.items():
        try:
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.endswith('.csv')]
                symbols = [f.replace('.csv', '') for f in files]
                all_symbols.update(symbols)
                print(f"Found {len(symbols)} symbols in {data_type}")
            else:
                print(f"Path not found: {path}")
        except Exception as e:
            print(f"Error reading {data_type}: {e}")

    return sorted(list(all_symbols))

def load_symbol_data(symbol):
    """Load 15-minute data for symbol"""
    try:
        file_path = os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv")
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)

        if len(df) < 500:  # Need minimum data
            return None

        return df
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return None

def add_technical_indicators(df):
    """Add the proven technical indicators"""
    try:
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        # ADX
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Price indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['high_20'] = df['high'].rolling(20).max()

        # Fill NaN values
        numeric_columns = ['rsi', 'adx', 'volume_ratio', 'sma_20', 'high_20', 'volume_sma']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        return df
    except Exception as e:
        print(f"Error adding indicators: {e}")
        return df

def generate_trading_signals(df):
    """Generate trading signals using proven logic"""
    try:
        df['signal'] = 0

        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            # PROVEN signal conditions (same as ₹894 system)
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

            # Require 4 out of 6 conditions (PROVEN OPTIMAL)
            if sum(conditions) >= 4:
                df.iloc[i, df.columns.get_loc('signal')] = 1

        return df
    except Exception as e:
        print(f"Error generating signals: {e}")
        df['signal'] = 0
        return df

def backtest_symbol(df, symbol):
    """Backtest using proven parameters"""
    try:
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

                # PROVEN exit conditions
                exit_reason = None
                if current_return >= PROFIT_TARGET:
                    exit_reason = 'Profit Target'
                elif current_return <= -STOP_LOSS:
                    exit_reason = 'Stop Loss'
                elif bars_held >= MAX_HOLD_BARS:
                    exit_reason = 'Time Exit'

                if exit_reason:
                    shares = pos['shares']
                    exit_price = current['close'] * 0.9995  # Slippage

                    pnl = (exit_price - pos['entry_price']) * shares
                    commission = (pos['entry_price'] + exit_price) * shares * 0.0005
                    net_pnl = pnl - commission

                    trade = {
                        'symbol': symbol,
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
                len(positions) < MAX_POSITIONS and
                cash > 5000):

                # PROVEN position sizing
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
        print(f"Error in backtest for {symbol}: {e}")
        return []

def run_comprehensive_all_symbols_test():
    """Test ALL available symbols comprehensively"""

    print("🚀 COMPREHENSIVE ALL-SYMBOLS TESTING")
    print("=" * 50)
    print("Testing EVERY symbol in your data directories!")
    print()

    # Get all available symbols
    all_symbols = get_all_available_symbols()

    if not all_symbols:
        print("No symbols found in data directories!")
        return

    print(f"\n📊 FOUND {len(all_symbols)} TOTAL SYMBOLS")
    print(f"Testing with PROVEN ₹894 system parameters...")
    print()

    all_results = []
    successful_symbols = 0
    failed_symbols = 0

    # Test each symbol
    for i, symbol in enumerate(all_symbols, 1):
        print(f"[{i}/{len(all_symbols)}] Testing {symbol}...")

        try:
            # Load data
            df = load_symbol_data(symbol)
            if df is None:
                print(f"  ✗ No/insufficient data")
                failed_symbols += 1
                continue

            # Add indicators
            df = add_technical_indicators(df)

            # Generate signals
            df = generate_trading_signals(df)

            # Check signal count
            signal_count = df['signal'].sum()
            if signal_count < 5:
                print(f"  ✗ Only {signal_count} signals")
                failed_symbols += 1
                continue

            print(f"  📊 {signal_count} signals generated")

            # Run backtest
            trades = backtest_symbol(df, symbol)

            if len(trades) >= 3:  # Need minimum trades
                # Calculate comprehensive stats
                df_trades = pd.DataFrame(trades)
                winning_trades = len(df_trades[df_trades['pnl'] > 0])

                result = {
                    'symbol': symbol,
                    'total_trades': len(trades),
                    'winning_trades': winning_trades,
                    'losing_trades': len(trades) - winning_trades,
                    'win_rate': (winning_trades / len(trades)) * 100,
                    'total_pnl': df_trades['pnl'].sum(),
                    'avg_pnl_per_trade': df_trades['pnl'].mean(),
                    'best_trade': df_trades['pnl'].max(),
                    'worst_trade': df_trades['pnl'].min(),
                    'avg_return_pct': df_trades['return_pct'].mean() * 100,
                    'avg_hold_bars': df_trades['bars_held'].mean(),
                    'signals_generated': signal_count,
                    'profit_factor': abs(df_trades[df_trades['pnl'] > 0]['pnl'].sum() / 
                                       df_trades[df_trades['pnl'] <= 0]['pnl'].sum()) if len(df_trades[df_trades['pnl'] <= 0]) > 0 else float('inf'),
                    'max_drawdown': df_trades['pnl'].cumsum().min(),
                    'data_bars': len(df),
                    'is_profitable': df_trades['pnl'].sum() > 0
                }

                all_results.append(result)
                successful_symbols += 1

                status = "✅ PROFIT" if result['is_profitable'] else "❌ LOSS"
                print(f"  {status}: {result['total_trades']} trades, "
                      f"{result['win_rate']:.1f}% win rate, "
                      f"₹{result['total_pnl']:.0f} PnL")
            else:
                print(f"  ✗ Only {len(trades)} trades")
                failed_symbols += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_symbols += 1
            continue

    # COMPREHENSIVE ANALYSIS
    if all_results:
        results_df = pd.DataFrame(all_results)

        print(f"\n\n🏆 COMPREHENSIVE ALL-SYMBOLS ANALYSIS")
        print("=" * 55)

        # Overall statistics
        total_symbols_tested = len(all_symbols)
        successful_count = len(results_df)
        profitable_count = len(results_df[results_df['is_profitable']])
        losing_count = successful_count - profitable_count

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total Symbols Available: {total_symbols_tested}")
        print(f"  Successfully Tested: {successful_count} ({successful_count/total_symbols_tested*100:.1f}%)")
        print(f"  Failed Testing: {failed_symbols} ({failed_symbols/total_symbols_tested*100:.1f}%)")
        print(f"  Profitable Symbols: {profitable_count} ({profitable_count/successful_count*100:.1f}% of tested)")
        print(f"  Losing Symbols: {losing_count} ({losing_count/successful_count*100:.1f}% of tested)")

        # Performance metrics
        total_trades = results_df['total_trades'].sum()
        total_pnl = results_df['total_pnl'].sum()
        avg_win_rate = results_df['win_rate'].mean()

        print(f"\n💰 SYSTEM PERFORMANCE:")
        print(f"  Total Trades: {total_trades:,}")
        print(f"  Total PnL: ₹{total_pnl:,.0f}")
        print(f"  Average Win Rate: {avg_win_rate:.1f}%")
        print(f"  Return on Capital: {(total_pnl/INITIAL_CAPITAL)*100:.2f}%")

        # Top and bottom performers
        results_df_sorted = results_df.sort_values('total_pnl', ascending=False)

        print(f"\n🏆 TOP 10 PROFITABLE SYMBOLS:")
        print(f"{'Symbol':<12} {'Trades':<7} {'Win Rate':<9} {'Total PnL':<11} {'Profit Factor'}")
        print("-" * 65)

        top_10 = results_df_sorted.head(10)
        for _, row in top_10.iterrows():
            pf = row['profit_factor'] if row['profit_factor'] != float('inf') else 99.99
            print(f"{row['symbol']:<12} {row['total_trades']:<7} {row['win_rate']:<9.1f}% "
                  f"₹{row['total_pnl']:<10.0f} {pf:<6.2f}")

        print(f"\n💸 WORST 10 PERFORMING SYMBOLS:")
        print(f"{'Symbol':<12} {'Trades':<7} {'Win Rate':<9} {'Total PnL':<11} {'Profit Factor'}")
        print("-" * 65)

        bottom_10 = results_df_sorted.tail(10)
        for _, row in bottom_10.iterrows():
            pf = row['profit_factor'] if row['profit_factor'] != float('inf') else 99.99
            print(f"{row['symbol']:<12} {row['total_trades']:<7} {row['win_rate']:<9.1f}% "
                  f"₹{row['total_pnl']:<10.0f} {pf:<6.2f}")

        # Comparison with original 6 symbols
        original_6_symbols = ['FINPIPE', 'GREENLAM', 'WABAG', 'ITI', 'LTTS', 'CONCORDBIO']
        original_results = results_df[results_df['symbol'].isin(original_6_symbols)]

        if len(original_results) > 0:
            original_pnl = original_results['total_pnl'].sum()
            original_trades = original_results['total_trades'].sum()

            print(f"\n🔍 COMPARISON WITH ORIGINAL 6 SYMBOLS:")
            print(f"  Original 6 Symbols PnL: ₹{original_pnl:.0f} ({original_trades} trades)")
            print(f"  All Symbols PnL: ₹{total_pnl:.0f} ({total_trades} trades)")
            print(f"  Improvement: {((total_pnl - original_pnl) / abs(original_pnl)) * 100 if original_pnl != 0 else 0:+.1f}%")

        # Distribution analysis
        print(f"\n📈 PERFORMANCE DISTRIBUTION:")
        profitable_symbols = results_df[results_df['is_profitable']]
        losing_symbols = results_df[~results_df['is_profitable']]

        if len(profitable_symbols) > 0:
            print(f"  Profitable Symbols ({len(profitable_symbols)}):")
            print(f"    Average PnL: ₹{profitable_symbols['total_pnl'].mean():.0f}")
            print(f"    Total Contribution: ₹{profitable_symbols['total_pnl'].sum():.0f}")
            print(f"    Best Performer: {profitable_symbols.loc[profitable_symbols['total_pnl'].idxmax(), 'symbol']} "
                  f"(₹{profitable_symbols['total_pnl'].max():.0f})")

        if len(losing_symbols) > 0:
            print(f"  Losing Symbols ({len(losing_symbols)}):")
            print(f"    Average Loss: ₹{losing_symbols['total_pnl'].mean():.0f}")
            print(f"    Total Drain: ₹{losing_symbols['total_pnl'].sum():.0f}")
            print(f"    Worst Performer: {losing_symbols.loc[losing_symbols['total_pnl'].idxmin(), 'symbol']} "
                  f"(₹{losing_symbols['total_pnl'].min():.0f})")

        # Save comprehensive results
        results_df.to_csv('all_symbols_comprehensive_results.csv', index=False)
        print(f"\n💾 Results saved to: all_symbols_comprehensive_results.csv")

        # Final recommendation
        print(f"\n🎯 FINAL ANALYSIS:")
        if total_pnl > 0:
            print(f"✅ SYSTEM IS PROFITABLE across all symbols!")
            print(f"   Total profit: ₹{total_pnl:,.0f}")
            print(f"   Success rate: {profitable_count}/{successful_count} symbols")
            print(f"   Recommended for live trading")
        else:
            print(f"❌ System shows overall loss across all symbols")
            print(f"   Total loss: ₹{total_pnl:,.0f}")
            print(f"   Consider filtering to only profitable symbols")

        return results_df
    else:
        print("\n❌ No successful results across any symbols")
        return None

if __name__ == "__main__":
    results = run_comprehensive_all_symbols_test()
