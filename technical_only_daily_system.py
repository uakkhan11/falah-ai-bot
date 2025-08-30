
import os
import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# TECHNICAL-ONLY DAILY TRADING SYSTEM (NO ML)
# =============================================================================

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data")
}

# Realistic parameters for technical-only trading
INITIAL_CAPITAL = 100000
MAX_POSITIONS = 3
POSITION_SIZE = 0.02  # 2% of capital per trade
YEAR_FILTER = 2025

# Technical signal parameters
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
ADX_MIN = 20
VOLUME_MULT = 1.5

# Exit parameters  
PROFIT_TARGET = 0.025   # 2.5% profit target
STOP_LOSS = 0.015       # 1.5% stop loss
MAX_HOLD_BARS = 25      # Max hold time

def load_data(symbol):
    """Load 15-minute data"""
    try:
        df = pd.read_csv(os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"))
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)

        if len(df) < 500:
            return None

        return df
    except:
        return None

def add_technical_indicators(df):
    """Add simple technical indicators"""
    try:
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        # ADX for trend strength
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

def generate_technical_signals(df):
    """Generate pure technical signals (no ML)"""
    try:
        df['signal'] = 0

        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            # Multi-condition technical signal
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

            # Require at least 4 out of 6 conditions
            if sum(conditions) >= 4:
                df.iloc[i, df.columns.get_loc('signal')] = 1

        return df
    except Exception as e:
        print(f"Error generating signals: {e}")
        df['signal'] = 0
        return df

def technical_backtest(df, symbol):
    """Simple technical backtesting"""
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

                # Exit conditions
                exit_reason = None
                if current_return >= PROFIT_TARGET:
                    exit_reason = 'Profit Target'
                elif current_return <= -STOP_LOSS:
                    exit_reason = 'Stop Loss'
                elif bars_held >= MAX_HOLD_BARS:
                    exit_reason = 'Time Exit'

                if exit_reason:
                    # Calculate trade
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

                # Fixed position sizing
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
        print(f"Error in backtest: {e}")
        return []

def run_technical_only_backtest():
    """Run technical-only backtesting system"""

    print("🔧 TECHNICAL-ONLY DAILY TRADING SYSTEM")
    print("=" * 50)
    print("No ML - Pure Technical Signals Only!")
    print()
    print("Parameters:")
    print(f"  Position Size: {POSITION_SIZE*100}% of capital")
    print(f"  Profit Target: {PROFIT_TARGET*100}%")
    print(f"  Stop Loss: {STOP_LOSS*100}%")
    print(f"  Max Hold: {MAX_HOLD_BARS} bars")
    print()

    # Get symbols
    try:
        symbols = [f.replace('.csv', '') for f in os.listdir(DATA_PATHS['daily']) 
                  if f.endswith('.csv')][:15]
        print(f"Testing {len(symbols)} symbols")
    except:
        print("Error loading symbols")
        return

    all_results = []
    successful_symbols = 0

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

        try:
            # Load data
            df = load_data(symbol)
            if df is None:
                print(f"  ✗ No data for {symbol}")
                continue

            # Add technical indicators
            df = add_technical_indicators(df)

            # Generate technical signals
            df = generate_technical_signals(df)

            # Check signal count
            signal_count = df['signal'].sum()
            if signal_count < 5:
                print(f"  ✗ Only {signal_count} signals")
                continue

            print(f"  📊 {signal_count} technical signals generated")

            # Run backtest
            trades = technical_backtest(df, symbol)

            if len(trades) >= 3:  # Need minimum trades
                # Calculate stats
                df_trades = pd.DataFrame(trades)
                winning_trades = len(df_trades[df_trades['pnl'] > 0])

                result = {
                    'symbol': symbol,
                    'total_trades': len(trades),
                    'winning_trades': winning_trades,
                    'win_rate': winning_trades / len(trades) * 100,
                    'total_pnl': df_trades['pnl'].sum(),
                    'avg_pnl_per_trade': df_trades['pnl'].mean(),
                    'best_trade': df_trades['pnl'].max(),
                    'worst_trade': df_trades['pnl'].min(),
                    'avg_return_pct': df_trades['return_pct'].mean() * 100,
                    'avg_hold_bars': df_trades['bars_held'].mean()
                }

                all_results.append(result)
                successful_symbols += 1

                print(f"  ✅ {result['total_trades']} trades, "
                      f"{result['win_rate']:.1f}% win rate, "
                      f"₹{result['total_pnl']:.0f} PnL")
            else:
                print(f"  ✗ Only {len(trades)} trades executed")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    # Display results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('total_pnl', ascending=False)

        print(f"\n\n🎯 TECHNICAL-ONLY RESULTS:")
        print("=" * 40)
        print(f"Successfully processed: {successful_symbols}/{len(symbols)} symbols")
        print()

        print("TOP PERFORMERS:")
        print("-" * 30)
        display_cols = ['symbol', 'total_trades', 'win_rate', 'total_pnl', 'avg_return_pct']
        print(results_df[display_cols].head(10).round(2).to_string(index=False))

        # Overall stats
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Trades: {results_df['total_trades'].sum()}")
        print(f"  Average Win Rate: {results_df['win_rate'].mean():.1f}%")
        print(f"  Total PnL: ₹{results_df['total_pnl'].sum():.0f}")
        print(f"  Average Return per Trade: {results_df['avg_return_pct'].mean():.2f}%")
        print(f"  Average Hold Time: {results_df['avg_hold_bars'].mean():.1f} bars")
        print(f"  Profitable Symbols: {len(results_df[results_df['total_pnl'] > 0])}")

        # Save results
        results_df.to_csv('technical_only_results.csv', index=False)
        print(f"\nResults saved to: technical_only_results.csv")

        return results_df
    else:
        print("\nNo successful results from technical-only approach.")
        return None

if __name__ == "__main__":
    results = run_technical_only_backtest()
