# enhanced_cnc_with_vwap_strategies.py

import pandas as pd
import numpy as np
import os
import json
import warnings
from datetime import datetime, timedelta
import gc

warnings.filterwarnings("ignore")

# ========================
# ENHANCED CNC COMPARISON WITH VWAP STRATEGIES
# ========================
BASE_DIR = "/root/falah-ai-bot/"
DATA_DIRS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    'results': os.path.join(BASE_DIR, "vwap_enhanced_comparison"),
}

os.makedirs(DATA_DIRS['results'], exist_ok=True)

# ENHANCED CNC STRATEGIES WITH VWAP
ENHANCED_CNC_STRATEGIES = {
    'daily_vwap_strategies': {
        'data_dir': DATA_DIRS['daily'],
        'timeframe': 'daily',
        'strategies': {
            'vwap_mean_reversion_cnc': {
                'name': 'VWAP Mean Reversion CNC (Daily)',
                'profit_target': 0.12,          # 12%
                'stop_loss': 0.05,              # 5%
                'vwap_deviation_buy': -0.015,   # Buy when 1.5% below VWAP
                'vwap_deviation_sell': 0.010,   # Sell when 1% above VWAP
                'min_hold_days': 1,
                'max_hold_days': 18,
                'trailing_trigger': 0.06,
                'trailing_distance': 0.025
            },
            'vwap_breakout_cnc': {
                'name': 'VWAP Breakout CNC (Daily)',
                'profit_target': 0.15,          # 15%
                'stop_loss': 0.06,              # 6%
                'vwap_breakout_threshold': 0.008, # Buy when breaking 0.8% above VWAP
                'volume_confirmation': 1.2,      # Require 20% above average volume
                'min_hold_days': 1,
                'max_hold_days': 25,
                'trailing_trigger': 0.08,
                'trailing_distance': 0.03
            },
            'vwap_confluence_cnc': {
                'name': 'VWAP + Williams %R Confluence CNC',
                'profit_target': 0.14,          # 14%
                'stop_loss': 0.055,             # 5.5%
                'vwap_deviation_threshold': -0.012, # 1.2% below VWAP
                'williams_r_oversold': -80,
                'min_hold_days': 1,
                'max_hold_days': 20,
                'trailing_trigger': 0.07,
                'trailing_distance': 0.028
            },
            'multi_timeframe_vwap_cnc': {
                'name': 'Multi-Timeframe VWAP CNC',
                'profit_target': 0.13,          # 13%
                'stop_loss': 0.05,              # 5%
                'daily_vwap_bias': True,        # Overall bias from daily VWAP
                'entry_deviation': -0.010,      # Entry when 1% below VWAP
                'min_hold_days': 1,
                'max_hold_days': 16,
                'trailing_trigger': 0.065,
                'trailing_distance': 0.025
            }
        }
    },
    'intraday_vwap_timing_cnc': {
        'data_dir': DATA_DIRS['15minute'],
        'timeframe': '15minute',
        'strategies': {
            'precision_vwap_cnc': {
                'name': 'Precision VWAP Entry CNC (15min timing)',
                'profit_target': 0.08,          # 8%
                'stop_loss': 0.04,              # 4%
                'vwap_entry_deviation': -0.008, # Enter when 0.8% below VWAP
                'volume_filter': 1.1,           # Above average volume
                'min_hold_days': 1,             # CNC requirement
                'max_hold_days': 5,
                'trailing_trigger': 0.04,
                'trailing_distance': 0.02
            },
            'vwap_volume_profile_cnc': {
                'name': 'VWAP Volume Profile CNC (15min timing)',
                'profit_target': 0.10,          # 10%
                'stop_loss': 0.045,             # 4.5%
                'volume_spike_threshold': 1.5,  # 50% above average
                'vwap_distance_threshold': 0.006, # Within 0.6% of VWAP
                'min_hold_days': 1,             # CNC requirement
                'max_hold_days': 7,
                'trailing_trigger': 0.05,
                'trailing_distance': 0.025
            }
        }
    }
}

# Keep existing successful strategies for comparison
PROVEN_WINNERS = {
    'williams_r_cnc': {
        'name': 'Williams %R CNC (Daily) - CHAMPION',
        'profit_target': 0.12,
        'stop_loss': 0.05,
        'oversold_threshold': -80,
        'overbought_threshold': -20,
        'min_hold_days': 1,
        'max_hold_days': 20,
        'trailing_trigger': 0.06,
        'trailing_distance': 0.025
    },
    'adx_trend_cnc': {
        'name': 'ADX Trend CNC (Daily) - RUNNER-UP',
        'profit_target': 0.18,
        'stop_loss': 0.07,
        'adx_threshold': 25,
        'min_hold_days': 1,
        'max_hold_days': 30,
        'trailing_trigger': 0.10,
        'trailing_distance': 0.035
    }
}

# ZERODHA CNC CHARGES
ZERODHA_CNC_CHARGES = {
    'brokerage_rate': 0.0000,
    'stt_buy_rate': 0.001,
    'stt_sell_rate': 0.001,
    'exchange_txn_rate': 0.0000345,
    'gst_rate': 0.18,
    'sebi_rate': 0.000001,
    'stamp_duty_rate': 0.00015,
    'dp_charges_per_sell': 13.5
}

INITIAL_CAPITAL = 1000000
POSITION_SIZE_PER_TRADE = 100000

class VWAPEnhancedComparison:
    def __init__(self):
        self.results_summary = {}
        self.execution_stats = {
            'total_strategies_tested': 0,
            'successful_strategies': 0,
            'vwap_strategies_tested': 0,
            'total_trades_analyzed': 0,
            'start_time': datetime.now()
        }
        
        print("📊 VWAP-Enhanced CNC Strategy Comparison Initialized")
        print("✅ Adding popular VWAP strategies to the mix")
        print("🔄 Testing against proven Williams %R champion")
    
    def _calculate_enhanced_vwap(self, df):
        """Calculate comprehensive VWAP variants"""
        try:
            print(f"       📊 Calculating enhanced VWAP indicators...")
            
            # Check for volume data
            if 'volume' not in df.columns or df['volume'].sum() == 0:
                print(f"       ⚠️ No volume data - using price-based VWAP approximation")
                # Fallback: Use typical price as proxy
                df['volume'] = 1  # Equal weighting
            
            # Standard VWAP calculation
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Ensure no zero volumes
            df['volume'] = df['volume'].replace(0, 1)
            
            # Calculate cumulative VWAP (resets daily for intraday, rolling for daily)
            if len(df) > 1:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    # Group by date for daily reset VWAP
                    df['vwap'] = df.groupby(df['date'].dt.date).apply(
                        lambda x: (x['volume'] * typical_price.loc[x.index]).cumsum() / x['volume'].cumsum()
                    ).values
                else:
                    # Rolling VWAP for daily data
                    volume_price = typical_price * df['volume']
                    df['vwap'] = volume_price.cumsum() / df['volume'].cumsum()
            
            # VWAP deviation
            df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
            df['vwap_deviation_pct'] = df['vwap_deviation'] * 100
            
            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_spike'] = df['volume_ratio'] > 1.2  # 20% above average
            
            # VWAP bands (similar to Bollinger Bands)
            vwap_std = df['vwap_deviation'].rolling(20).std()
            df['vwap_upper'] = df['vwap'] * (1 + vwap_std * 1.5)
            df['vwap_lower'] = df['vwap'] * (1 - vwap_std * 1.5)
            
            # VWAP trend
            df['vwap_trend'] = df['vwap'].rolling(5).mean()
            df['vwap_rising'] = df['vwap'] > df['vwap_trend']
            
            print(f"       ✅ VWAP indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"       ❌ VWAP calculation failed: {e}")
            # Fallback VWAP
            df['vwap'] = df['close'].rolling(20).mean()
            df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
            return df
    
    def _calculate_supporting_indicators(self, df):
        """Calculate supporting indicators for confluence strategies"""
        try:
            # Williams %R for confluence
            if 'williams_r' not in df.columns:
                highest_high = df['high'].rolling(14).max()
                lowest_low = df['low'].rolling(14).min()
                df['williams_r'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
            
            # ADX for trend confirmation
            if len(df) > 30:
                df['tr1'] = df['high'] - df['low']
                df['tr2'] = abs(df['high'] - df['close'].shift(1))
                df['tr3'] = abs(df['low'] - df['close'].shift(1))
                df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
                df['atr'] = df['tr'].rolling(14).mean()
                
                df['plus_dm'] = np.where(
                    (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                    np.maximum(df['high'] - df['high'].shift(1), 0), 0
                )
                df['minus_dm'] = np.where(
                    (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                    np.maximum(df['low'].shift(1) - df['low'], 0), 0
                )
                
                df['plus_di'] = (df['plus_dm'].rolling(14).mean() / df['atr']) * 100
                df['minus_di'] = (df['minus_dm'].rolling(14).mean() / df['atr']) * 100
                dx = abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']) * 100
                df['adx'] = dx.rolling(14).mean()
            
            return df
            
        except Exception as e:
            print(f"       ⚠️ Supporting indicators calculation failed: {e}")
            return df
    
    def _generate_vwap_signals(self, df, strategy_key, strategy_config):
        """Generate VWAP-based trading signals"""
        try:
            df['signal'] = 0
            
            if 'vwap_mean_reversion' in strategy_key:
                # Mean reversion around VWAP
                buy_threshold = strategy_config['vwap_deviation_buy']
                sell_threshold = strategy_config['vwap_deviation_sell']
                
                buy_condition = (df['vwap_deviation'] < buy_threshold) & (df['close'] > df['close'].shift(1))
                sell_condition = (df['vwap_deviation'] > sell_threshold) | (df['close'] < df['vwap'])
                
            elif 'vwap_breakout' in strategy_key:
                # VWAP breakout with volume confirmation
                breakout_threshold = strategy_config['vwap_breakout_threshold']
                volume_confirmation = strategy_config['volume_confirmation']
                
                buy_condition = (
                    (df['vwap_deviation'] > breakout_threshold) & 
                    (df['volume_ratio'] > volume_confirmation) &
                    (df['close'] > df['vwap'])
                )
                sell_condition = df['close'] < df['vwap'] * 0.995  # Exit below VWAP
                
            elif 'vwap_confluence' in strategy_key:
                # VWAP + Williams %R confluence
                vwap_threshold = strategy_config['vwap_deviation_threshold']
                williams_oversold = strategy_config['williams_r_oversold']
                
                buy_condition = (
                    (df['vwap_deviation'] < vwap_threshold) & 
                    (df['williams_r'] < williams_oversold) &
                    (df['williams_r'] > df['williams_r'].shift(1))  # Turning up
                )
                sell_condition = (df['vwap_deviation'] > 0.01) | (df['williams_r'] > -20)
                
            elif 'multi_timeframe_vwap' in strategy_key:
                # Multi-timeframe VWAP approach
                entry_deviation = strategy_config['entry_deviation']
                
                buy_condition = (
                    (df['vwap_deviation'] < entry_deviation) & 
                    (df['vwap_rising']) &  # VWAP trend rising
                    (df['close'] > df['close'].shift(1))
                )
                sell_condition = df['close'] < df['vwap'] * 0.99
                
            elif 'precision_vwap' in strategy_key:
                # Precision VWAP entry with volume filter
                entry_deviation = strategy_config['vwap_entry_deviation']
                volume_filter = strategy_config['volume_filter']
                
                buy_condition = (
                    (df['vwap_deviation'] < entry_deviation) & 
                    (df['volume_ratio'] > volume_filter) &
                    (df['close'] > df['close'].shift(1))
                )
                sell_condition = df['vwap_deviation'] > 0.005  # Exit when above VWAP
                
            elif 'vwap_volume_profile' in strategy_key:
                # VWAP with volume profile analysis
                volume_threshold = strategy_config['volume_spike_threshold']
                distance_threshold = strategy_config['vwap_distance_threshold']
                
                buy_condition = (
                    (abs(df['vwap_deviation']) < distance_threshold) & 
                    (df['volume_ratio'] > volume_threshold) &
                    (df['close'] > df['vwap'])
                )
                sell_condition = df['vwap_deviation'] > 0.008
                
            elif 'williams_r' in strategy_key:
                # Proven Williams %R strategy
                oversold = strategy_config['oversold_threshold']
                overbought = strategy_config['overbought_threshold']
                buy_condition = (df['williams_r'] < oversold) & (df['williams_r'] > df['williams_r'].shift(1))
                sell_condition = df['williams_r'] > overbought
                
            elif 'adx_trend' in strategy_key:
                # Proven ADX strategy
                if 'adx' in df.columns:
                    adx_threshold = strategy_config['adx_threshold']
                    buy_condition = (df['adx'] > adx_threshold) & (df['plus_di'] > df['minus_di'])
                    sell_condition = (df['adx'] < adx_threshold) | (df['plus_di'] < df['minus_di'])
                else:
                    buy_condition = pd.Series([False] * len(df), index=df.index)
                    sell_condition = pd.Series([False] * len(df), index=df.index)
            
            else:
                buy_condition = pd.Series([False] * len(df), index=df.index)
                sell_condition = pd.Series([False] * len(df), index=df.index)
            
            # Apply signals safely
            df.loc[buy_condition.fillna(False), 'signal'] = 1
            df.loc[sell_condition.fillna(False), 'signal'] = -1
            
            buy_signals = (df['signal'] == 1).sum()
            sell_signals = (df['signal'] == -1).sum()
            print(f"       📊 Generated {buy_signals} buy, {sell_signals} sell signals")
            
            return df
            
        except Exception as e:
            print(f"       ❌ VWAP signal generation failed: {e}")
            df['signal'] = 0
            return df
    
    def _calculate_cnc_charges(self, buy_value, sell_value):
        """Calculate CNC charges"""
        charges = ZERODHA_CNC_CHARGES
        
        stt = (buy_value * charges['stt_buy_rate']) + (sell_value * charges['stt_sell_rate'])
        exchange = (buy_value + sell_value) * charges['exchange_txn_rate']
        gst = exchange * charges['gst_rate']
        sebi = (buy_value + sell_value) * charges['sebi_rate']
        stamp_duty = buy_value * charges['stamp_duty_rate']
        dp_charges = charges['dp_charges_per_sell']
        
        total_charges = stt + exchange + gst + sebi + stamp_duty + dp_charges
        
        return {
            'total_charges': total_charges,
            'effective_cost_percentage': total_charges / buy_value * 100
        }
    
    def _backtest_cnc_strategy(self, df, strategy_name, strategy_config):
        """CNC backtesting with minimum 1-day holding"""
        try:
            trades = []
            portfolio_value = INITIAL_CAPITAL
            active_positions = {}
            position_id = 0
            
            profit_target = strategy_config['profit_target']
            stop_loss = strategy_config['stop_loss']
            min_hold_days = strategy_config.get('min_hold_days', 1)
            max_hold_days = strategy_config['max_hold_days']
            trailing_trigger = strategy_config.get('trailing_trigger', profit_target * 0.5)
            trailing_distance = strategy_config.get('trailing_distance', stop_loss * 0.5)
            max_positions = 10
            
            for i in range(1, len(df)):
                current_date = pd.to_datetime(df.iloc[i]['date' if 'date' in df.columns else 'datetime'])
                current_price = df.iloc[i]['close']
                signal = df.iloc[i]['signal']
                
                if not np.isfinite(current_price) or current_price <= 0:
                    continue
                
                # Check exits
                positions_to_close = []
                
                for pos_id, position in active_positions.items():
                    entry_date = position['entry_date']
                    entry_price = position['entry_price']
                    days_held = (current_date - entry_date).days
                    pct_change = (current_price - entry_price) / entry_price
                    
                    if current_price > position['highest_price']:
                        position['highest_price'] = current_price
                    
                    if not position['trailing_active'] and pct_change >= trailing_trigger:
                        position['trailing_active'] = True
                        position['trailing_stop'] = current_price * (1 - trailing_distance)
                    
                    if position['trailing_active']:
                        new_trailing_stop = current_price * (1 - trailing_distance)
                        if new_trailing_stop > position['trailing_stop']:
                            position['trailing_stop'] = new_trailing_stop
                    
                    should_exit = False
                    exit_reason = ""
                    
                    if days_held < min_hold_days:
                        continue
                    elif days_held >= max_hold_days:
                        should_exit = True
                        exit_reason = "Max Hold Period"
                    elif position['trailing_active'] and current_price <= position['trailing_stop']:
                        should_exit = True
                        exit_reason = "Trailing Stop"
                    elif pct_change <= -stop_loss:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif pct_change >= profit_target:
                        should_exit = True
                        exit_reason = "Profit Target"
                    elif signal == -1:
                        should_exit = True
                        exit_reason = "Signal Exit"
                    
                    if should_exit:
                        buy_value = POSITION_SIZE_PER_TRADE
                        sell_value = position['shares'] * current_price
                        charge_details = self._calculate_cnc_charges(buy_value, sell_value)
                        total_charges = charge_details['total_charges']
                        
                        gross_pnl = sell_value - buy_value
                        net_pnl = gross_pnl - total_charges
                        portfolio_value += net_pnl
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'shares': position['shares'],
                            'days_held': days_held,
                            'gross_pnl': gross_pnl,
                            'net_pnl': net_pnl,
                            'total_charges': total_charges,
                            'return_pct': pct_change * 100,
                            'exit_reason': exit_reason,
                            'trailing_activated': position['trailing_active'],
                            'portfolio_value': portfolio_value
                        })
                        
                        positions_to_close.append(pos_id)
                
                for pos_id in positions_to_close:
                    del active_positions[pos_id]
                
                # Entry logic
                if (signal == 1 and 
                    len(active_positions) < max_positions and
                    portfolio_value >= POSITION_SIZE_PER_TRADE * 1.5):
                    
                    position_id += 1
                    shares = POSITION_SIZE_PER_TRADE / current_price
                    
                    active_positions[position_id] = {
                        'entry_date': current_date,
                        'entry_price': current_price,
                        'shares': shares,
                        'highest_price': current_price,
                        'trailing_active': False,
                        'trailing_stop': 0
                    }
                    
                    portfolio_value -= POSITION_SIZE_PER_TRADE
            
            return trades, portfolio_value
            
        except Exception as e:
            print(f"       ❌ Backtest failed: {e}")
            return [], INITIAL_CAPITAL
    
    def _analyze_performance(self, trades, final_portfolio, strategy_name):
        """Analyze strategy performance"""
        if not trades:
            return None
        
        df_trades = pd.DataFrame(trades)
        
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['net_pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_return = (final_portfolio - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        avg_return_per_trade = df_trades['return_pct'].mean()
        avg_hold_days = df_trades['days_held'].mean()
        
        total_charges = df_trades['total_charges'].sum()
        avg_charges_per_trade = total_charges / total_trades if total_trades > 0 else 0
        
        total_gross_pnl = df_trades['gross_pnl'].sum()
        total_net_pnl = df_trades['net_pnl'].sum()
        charges_impact = (total_gross_pnl - total_net_pnl) / abs(total_gross_pnl) * 100 if total_gross_pnl != 0 else 0
        
        std_return = df_trades['return_pct'].std()
        sharpe_ratio = avg_return_per_trade / std_return if std_return > 0 else 0
        
        portfolio_values = df_trades['portfolio_value'].values
        max_drawdown = 0
        if len(portfolio_values) > 0:
            peak = INITIAL_CAPITAL
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'strategy_name': strategy_name,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_portfolio': final_portfolio,
            'avg_return_per_trade': avg_return_per_trade,
            'avg_hold_days': avg_hold_days,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_charges': total_charges,
            'avg_charges_per_trade': avg_charges_per_trade,
            'charges_impact_pct': charges_impact,
            'status': 'COMPLETED'
        }
    
    def run_vwap_enhanced_comparison(self):
        """Run comprehensive comparison including VWAP strategies"""
        print(f"\n📊 VWAP-ENHANCED CNC STRATEGY COMPARISON")
        print(f"{'='*70}")
        print(f"✅ Testing popular VWAP strategies in CNC mode")
        print(f"🏆 Comparing against proven Williams %R champion")
        print(f"💰 Initial Capital: ₹{INITIAL_CAPITAL:,}")
        print(f"{'='*70}")
        
        all_results = {}
        
        # Test VWAP strategies
        for group_key, group_config in ENHANCED_CNC_STRATEGIES.items():
            print(f"\n📈 PROCESSING {group_key.upper()}")
            print(f"{'='*50}")
            
            data_dir = group_config['data_dir']
            timeframe = group_config['timeframe']
            
            try:
                data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                print(f"📁 Found {len(data_files)} data files")
            except:
                print(f"❌ Data directory not found: {data_dir}")
                continue
            
            group_results = {}
            test_files = data_files[:15]  # Test 15 symbols
            
            for strategy_key, strategy_config in group_config['strategies'].items():
                print(f"\n🎯 VWAP STRATEGY: {strategy_config['name']}")
                self.execution_stats['vwap_strategies_tested'] += 1
                
                all_trades = []
                
                for i, filename in enumerate(test_files, 1):
                    try:
                        symbol = filename.replace('.csv', '')
                        print(f"   📊 [{i}/{len(test_files)}] {symbol}")
                        
                        file_path = os.path.join(data_dir, filename)
                        df = pd.read_csv(file_path)
                        
                        if len(df) < 100:
                            print(f"       ⚠️ Insufficient data ({len(df)} bars)")
                            continue
                        
                        if 'datetime' not in df.columns and 'date' in df.columns:
                            df['datetime'] = pd.to_datetime(df['date'])
                        
                        # Calculate VWAP indicators
                        df = self._calculate_enhanced_vwap(df)
                        df = self._calculate_supporting_indicators(df)
                        
                        # Generate VWAP signals
                        df = self._generate_vwap_signals(df, strategy_key, strategy_config)
                        
                        # Run CNC backtest
                        symbol_trades, symbol_portfolio = self._backtest_cnc_strategy(
                            df, strategy_config['name'], strategy_config
                        )
                        
                        if symbol_trades:
                            for trade in symbol_trades:
                                trade['symbol'] = symbol
                            all_trades.extend(symbol_trades)
                            print(f"       ✅ {len(symbol_trades)} VWAP trades")
                        
                        del df
                        gc.collect()
                        
                    except Exception as e:
                        print(f"       ❌ Error: {e}")
                        continue
                
                if all_trades:
                    final_portfolio = INITIAL_CAPITAL + sum([trade['net_pnl'] for trade in all_trades])
                    
                    performance = self._analyze_performance(
                        all_trades, final_portfolio, strategy_config['name']
                    )
                    
                    if performance:
                        group_results[strategy_key] = performance
                        self.execution_stats['successful_strategies'] += 1
                        self.execution_stats['total_trades_analyzed'] += len(all_trades)
                        
                        print(f"   ✅ VWAP Analysis Complete:")
                        print(f"       📊 Trades: {performance['total_trades']}")
                        print(f"       🎯 Win Rate: {performance['win_rate']:.2%}")
                        print(f"       📈 Total Return: {performance['total_return']:.2f}%")
                        print(f"       📅 Avg Hold: {performance['avg_hold_days']:.1f} days")
                else:
                    print(f"   ❌ No VWAP trades generated")
            
            all_results[group_key] = group_results
        
        # Test proven winners for comparison
        print(f"\n🏆 TESTING PROVEN WINNERS FOR COMPARISON")
        print(f"{'='*50}")
        
        winners_results = {}
        daily_files = [f for f in os.listdir(DATA_DIRS['daily']) if f.endswith('.csv')][:15]
        
        for strategy_key, strategy_config in PROVEN_WINNERS.items():
            print(f"\n👑 CHAMPION: {strategy_config['name']}")
            
            all_trades = []
            
            for i, filename in enumerate(daily_files, 1):
                try:
                    symbol = filename.replace('.csv', '')
                    print(f"   📊 [{i}/{len(daily_files)}] {symbol}")
                    
                    file_path = os.path.join(DATA_DIRS['daily'], filename)
                    df = pd.read_csv(file_path)
                    
                    if len(df) < 100:
                        continue
                    
                    if 'datetime' not in df.columns and 'date' in df.columns:
                        df['datetime'] = pd.to_datetime(df['date'])
                    
                    df = self._calculate_supporting_indicators(df)
                    df = self._generate_vwap_signals(df, strategy_key, strategy_config)
                    
                    symbol_trades, _ = self._backtest_cnc_strategy(
                        df, strategy_config['name'], strategy_config
                    )
                    
                    if symbol_trades:
                        for trade in symbol_trades:
                            trade['symbol'] = symbol
                        all_trades.extend(symbol_trades)
                        print(f"       ✅ {len(symbol_trades)} champion trades")
                    
                    del df
                    gc.collect()
                    
                except Exception as e:
                    continue
            
            if all_trades:
                final_portfolio = INITIAL_CAPITAL + sum([trade['net_pnl'] for trade in all_trades])
                performance = self._analyze_performance(
                    all_trades, final_portfolio, strategy_config['name']
                )
                
                if performance:
                    winners_results[strategy_key] = performance
                    print(f"   ✅ Champion confirmed: {performance['total_return']:.2f}% return")
        
        all_results['proven_champions'] = winners_results
        
        self.results_summary = all_results
        self._print_vwap_comparison()
        
        return all_results
    
    def _print_vwap_comparison(self):
        """Print VWAP-enhanced comparison results"""
        print(f"\n{'='*80}")
        print(f"📊 VWAP-ENHANCED CNC STRATEGY COMPARISON RESULTS")
        print(f"{'='*80}")
        
        print(f"🔄 VWAP Strategies Tested: {self.execution_stats['vwap_strategies_tested']}")
        print(f"✅ Successful Strategies: {self.execution_stats['successful_strategies']}")
        print(f"📈 Total Trades: {self.execution_stats['total_trades_analyzed']}")
        
        # Collect all strategies
        all_strategies = []
        vwap_strategies = []
        proven_strategies = []
        
        for group_key, strategies in self.results_summary.items():
            for strategy_key, performance in strategies.items():
                if performance and performance.get('status') == 'COMPLETED':
                    all_strategies.append(performance)
                    
                    if 'VWAP' in performance['strategy_name']:
                        vwap_strategies.append(performance)
                    elif 'CHAMPION' in performance['strategy_name'] or 'RUNNER-UP' in performance['strategy_name']:
                        proven_strategies.append(performance)
        
        if all_strategies:
            all_strategies.sort(key=lambda x: x['total_return'], reverse=True)
            vwap_strategies.sort(key=lambda x: x['total_return'], reverse=True)
            proven_strategies.sort(key=lambda x: x['total_return'], reverse=True)
            
            print(f"\n🏆 OVERALL RANKING (ALL STRATEGIES):")
            print(f"{'='*60}")
            
            for i, strategy in enumerate(all_strategies, 1):
                strategy_type = "📊 VWAP" if "VWAP" in strategy['strategy_name'] else "👑 PROVEN"
                
                print(f"{i}. {strategy_type} {strategy['strategy_name']}")
                print(f"   📈 Total Return: {strategy['total_return']:.2f}%")
                print(f"   🎯 Win Rate: {strategy['win_rate']:.2%}")
                print(f"   📊 Trades: {strategy['total_trades']}")
                print(f"   📅 Avg Hold: {strategy['avg_hold_days']:.1f} days")
                print(f"   📊 Sharpe: {strategy['sharpe_ratio']:.2f}")
                print()
            
            print(f"📊 VWAP STRATEGIES PERFORMANCE:")
            print(f"{'='*40}")
            
            if vwap_strategies:
                best_vwap = vwap_strategies[0]
                print(f"🏆 Best VWAP Strategy: {best_vwap['strategy_name']}")
                print(f"   📈 Return: {best_vwap['total_return']:.2f}%")
                print(f"   🎯 Win Rate: {best_vwap['win_rate']:.2%}")
                print()
                
                for vwap in vwap_strategies:
                    print(f"• {vwap['strategy_name']}: {vwap['total_return']:.2f}% return")
            else:
                print("❌ No successful VWAP strategies found")
            
            print(f"\n👑 PROVEN CHAMPIONS PERFORMANCE:")
            print(f"{'='*40}")
            
            for proven in proven_strategies:
                print(f"👑 {proven['strategy_name']}: {proven['total_return']:.2f}% return")
            
            # Head-to-head comparison
            if vwap_strategies and proven_strategies:
                best_vwap = vwap_strategies
                champion = proven_strategies
                
                print(f"\n⚔️ HEAD-TO-HEAD: BEST VWAP vs CHAMPION")
                print(f"{'='*50}")
                print(f"📊 VWAP Champion: {best_vwap['strategy_name']}")
                print(f"   Return: {best_vwap['total_return']:.2f}% | Win Rate: {best_vwap['win_rate']:.2%}")
                print(f"👑 Proven Champion: {champion['strategy_name']}")
                print(f"   Return: {champion['total_return']:.2f}% | Win Rate: {champion['win_rate']:.2%}")
                
                winner = champion if champion['total_return'] > best_vwap['total_return'] else best_vwap
                margin = abs(champion['total_return'] - best_vwap['total_return'])
                
                print(f"\n🏆 WINNER: {winner['strategy_name']}")
                print(f"📈 Victory Margin: {margin:.2f}%")
        
        print(f"\n🎯 VWAP ANALYSIS COMPLETE!")

if __name__ == "__main__":
    print("📊 VWAP-ENHANCED CNC STRATEGY COMPARISON")
    print("="*60)
    print("✅ Testing popular VWAP strategies")
    print("📊 Mean reversion, breakout, confluence approaches")
    print("🏆 Head-to-head with Williams %R champion")
    print("🕌 All strategies 100% halal CNC compliant")
    print("="*60)
    
    try:
        comparator = VWAPEnhancedComparison()
        results = comparator.run_vwap_enhanced_comparison()
        
        print(f"\n🎉 VWAP-ENHANCED COMPARISON COMPLETED!")
        print(f"📊 Now we know how VWAP strategies perform vs Williams %R!")
        print(f"🏆 The ultimate CNC strategy champion revealed!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
