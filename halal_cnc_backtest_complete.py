# comprehensive_intraday_analysis.py

import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
from datetime import datetime, timedelta
import gc
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ========================
# COMPREHENSIVE INTRADAY ANALYSIS CONFIGURATION
# ========================
BASE_DIR = "/root/falah-ai-bot/"
DATA_DIRS = {
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    'models': os.path.join(BASE_DIR, "models"),
    'results': os.path.join(BASE_DIR, "comprehensive_intraday_results"),
}

os.makedirs(DATA_DIRS['results'], exist_ok=True)

# COMPREHENSIVE INDICATOR STRATEGIES
COMPREHENSIVE_STRATEGIES = {
    '15minute': {
        'data_dir': DATA_DIRS['15minute'],
        'timeframe': '15minute',
        'min_hold_minutes': 15,  # Minimum 15 minutes (1 candle)
        'strategies': {
            'williams_r_scalping': {
                'name': 'Williams %R (15min)',
                'profit_target': 0.015,      # 1.5%
                'stop_loss': 0.008,          # 0.8%
                'oversold_threshold': -85,
                'overbought_threshold': -15,
                'max_hold_minutes': 120,
                'trailing_trigger': 0.008,
                'trailing_distance': 0.004
            },
            'rsi_scalping': {
                'name': 'RSI (15min)',
                'profit_target': 0.012,      # 1.2%
                'stop_loss': 0.006,          # 0.6%
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'max_hold_minutes': 90,
                'trailing_trigger': 0.006,
                'trailing_distance': 0.003
            },
            'macd_scalping': {
                'name': 'MACD (15min)',
                'profit_target': 0.018,      # 1.8%
                'stop_loss': 0.009,          # 0.9%
                'max_hold_minutes': 150,
                'trailing_trigger': 0.010,
                'trailing_distance': 0.005
            },
            'bollinger_scalping': {
                'name': 'Bollinger Bands (15min)',
                'profit_target': 0.014,      # 1.4%
                'stop_loss': 0.007,          # 0.7%
                'max_hold_minutes': 100,
                'trailing_trigger': 0.007,
                'trailing_distance': 0.0035
            },
            'stochastic_scalping': {
                'name': 'Stochastic (15min)',
                'profit_target': 0.013,      # 1.3%
                'stop_loss': 0.0065,         # 0.65%
                'stoch_oversold': 20,
                'stoch_overbought': 80,
                'max_hold_minutes': 110,
                'trailing_trigger': 0.0065,
                'trailing_distance': 0.00325
            },
            'ema_cross_scalping': {
                'name': 'EMA Crossover (15min)',
                'profit_target': 0.016,      # 1.6%
                'stop_loss': 0.008,          # 0.8%
                'ema_fast': 5,
                'ema_slow': 13,
                'max_hold_minutes': 130,
                'trailing_trigger': 0.008,
                'trailing_distance': 0.004
            },
            'vwap_scalping': {
                'name': 'VWAP Deviation (15min)',
                'profit_target': 0.011,      # 1.1%
                'stop_loss': 0.0055,         # 0.55%
                'deviation_threshold': 0.005, # 0.5% from VWAP
                'max_hold_minutes': 85,
                'trailing_trigger': 0.0055,
                'trailing_distance': 0.00275
            }
        }
    },
    '1hour': {
        'data_dir': DATA_DIRS['1hour'],
        'timeframe': '1hour',
        'min_hold_minutes': 60,  # Minimum 1 hour
        'strategies': {
            'williams_r_swing': {
                'name': 'Williams %R (1hr)',
                'profit_target': 0.035,      # 3.5%
                'stop_loss': 0.015,          # 1.5%
                'oversold_threshold': -80,
                'overbought_threshold': -20,
                'max_hold_minutes': 360,
                'trailing_trigger': 0.020,
                'trailing_distance': 0.010
            },
            'rsi_swing': {
                'name': 'RSI (1hr)',
                'profit_target': 0.030,      # 3.0%
                'stop_loss': 0.012,          # 1.2%
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'max_hold_minutes': 300,
                'trailing_trigger': 0.015,
                'trailing_distance': 0.008
            },
            'macd_swing': {
                'name': 'MACD (1hr)',
                'profit_target': 0.040,      # 4.0%
                'stop_loss': 0.016,          # 1.6%
                'max_hold_minutes': 420,
                'trailing_trigger': 0.025,
                'trailing_distance': 0.012
            },
            'bollinger_swing': {
                'name': 'Bollinger Bands (1hr)',
                'profit_target': 0.032,      # 3.2%
                'stop_loss': 0.013,          # 1.3%
                'max_hold_minutes': 330,
                'trailing_trigger': 0.018,
                'trailing_distance': 0.009
            },
            'adx_trend_swing': {
                'name': 'ADX Trend (1hr)',
                'profit_target': 0.045,      # 4.5%
                'stop_loss': 0.018,          # 1.8%
                'adx_threshold': 25,
                'max_hold_minutes': 480,
                'trailing_trigger': 0.030,
                'trailing_distance': 0.015
            },
            'ema_cross_swing': {
                'name': 'EMA Crossover (1hr)',
                'profit_target': 0.028,      # 2.8%
                'stop_loss': 0.011,          # 1.1%
                'ema_fast': 9,
                'ema_slow': 21,
                'max_hold_minutes': 270,
                'trailing_trigger': 0.014,
                'trailing_distance': 0.007
            },
            'support_resistance_swing': {
                'name': 'Support/Resistance (1hr)',
                'profit_target': 0.038,      # 3.8%
                'stop_loss': 0.015,          # 1.5%
                'lookback_period': 20,
                'max_hold_minutes': 400,
                'trailing_trigger': 0.022,
                'trailing_distance': 0.011
            }
        }
    }
}

# TRADING PARAMETERS
INITIAL_CAPITAL = 1000000
POSITION_SIZE_PER_TRADE = 100000  # ₹1 lakh per trade
TRANSACTION_COST = 0.0015  # 0.15% for intraday

class ComprehensiveIntradayAnalyzer:
    def __init__(self):
        self.results_summary = {}
        self.trade_analysis = {}
        self.performance_comparison = {}
        self.execution_stats = {
            'total_strategies_tested': 0,
            'successful_strategies': 0,
            'failed_strategies': 0,
            'total_trades_analyzed': 0,
            'start_time': datetime.now()
        }
        
        print("🔍 Comprehensive Intraday Analyzer Initialized")
        print("📊 Testing 14 different indicators across 2 timeframes")
    
    def _calculate_comprehensive_indicators(self, df, timeframe):
        """Calculate all indicators for comparison"""
        try:
            print(f"       📊 Calculating comprehensive indicators...")
            
            # Ensure we have OHLCV data
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                print(f"       ⚠️ Missing OHLCV data")
                return df
            
            # Williams %R
            if 'williams_r' not in df.columns:
                highest_high = df['high'].rolling(14).max()
                lowest_low = df['low'].rolling(14).min()
                df['williams_r'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
            
            # RSI (if not already present)
            if 'rsi' not in df.columns:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp12 = df['close'].ewm(span=12).mean()
            exp26 = df['close'].ewm(span=26).mean()
            df['macd'] = exp12 - exp26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            df['bb_middle'] = df['close'].rolling(bb_period).mean()
            bb_std = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic
            lowest_low_14 = df['low'].rolling(14).min()
            highest_high_14 = df['high'].rolling(14).max()
            df['stoch_k'] = ((df['close'] - lowest_low_14) / (highest_high_14 - lowest_low_14)) * 100
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # EMA
            if timeframe == '15minute':
                df['ema_fast'] = df['close'].ewm(span=5).mean()
                df['ema_slow'] = df['close'].ewm(span=13).mean()
            else:  # 1hour
                df['ema_fast'] = df['close'].ewm(span=9).mean()
                df['ema_slow'] = df['close'].ewm(span=21).mean()
            
            # VWAP (if volume available)
            if 'volume' in df.columns and df['volume'].sum() > 0:
                if 'vwap' not in df.columns:
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    vwap_numerator = (typical_price * df['volume']).cumsum()
                    vwap_denominator = df['volume'].cumsum()
                    df['vwap'] = vwap_numerator / vwap_denominator
                
                df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
            
            # ADX (for 1-hour)
            if timeframe == '1hour':
                # Simplified ADX calculation
                df['tr'] = np.maximum(df['high'] - df['low'],
                                     np.maximum(abs(df['high'] - df['close'].shift(1)),
                                               abs(df['low'] - df['close'].shift(1))))
                df['atr'] = df['tr'].rolling(14).mean()
                
                # Directional movement
                df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                                        np.maximum(df['high'] - df['high'].shift(1), 0), 0)
                df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                                         np.maximum(df['low'].shift(1) - df['low'], 0), 0)
                
                df['plus_di'] = (df['plus_dm'].rolling(14).mean() / df['atr']) * 100
                df['minus_di'] = (df['minus_dm'].rolling(14).mean() / df['atr']) * 100
                df['adx'] = abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']) * 100
                df['adx'] = df['adx'].rolling(14).mean()
            
            # Support and Resistance levels
            lookback = 20
            df['resistance'] = df['high'].rolling(lookback).max()
            df['support'] = df['low'].rolling(lookback).min()
            df['sr_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
            
            print(f"       ✅ All indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"       ❌ Indicator calculation failed: {e}")
            return df
    
    def _generate_strategy_signals(self, df, strategy_key, strategy_config):
        """Generate signals for different strategies"""
        try:
            df['signal'] = 0  # Initialize signal column
            
            if 'williams' in strategy_key:
                oversold = strategy_config['oversold_threshold']
                overbought = strategy_config['overbought_threshold']
                # Buy: Williams %R oversold and turning up
                buy_condition = (df['williams_r'] < oversold) & (df['williams_r'] > df['williams_r'].shift(1))
                sell_condition = df['williams_r'] > overbought
                
            elif 'rsi' in strategy_key:
                oversold = strategy_config['rsi_oversold']
                overbought = strategy_config['rsi_overbought']
                buy_condition = (df['rsi'] < oversold) & (df['rsi'] > df['rsi'].shift(1))
                sell_condition = df['rsi'] > overbought
                
            elif 'macd' in strategy_key:
                # MACD crossover strategy
                buy_condition = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
                sell_condition = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
                
            elif 'bollinger' in strategy_key:
                # Bollinger Band mean reversion
                buy_condition = (df['bb_position'] < 0.2) & (df['close'] > df['close'].shift(1))  # Near lower band and turning up
                sell_condition = (df['bb_position'] > 0.8) | (df['close'] < df['bb_middle'])  # Near upper band or below middle
                
            elif 'stochastic' in strategy_key:
                oversold = strategy_config['stoch_oversold']
                overbought = strategy_config['stoch_overbought']
                buy_condition = (df['stoch_k'] < oversold) & (df['stoch_k'] > df['stoch_d'])
                sell_condition = df['stoch_k'] > overbought
                
            elif 'ema_cross' in strategy_key:
                # EMA crossover
                buy_condition = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
                sell_condition = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
                
            elif 'vwap' in strategy_key:
                if 'vwap_deviation' in df.columns:
                    threshold = strategy_config['deviation_threshold']
                    # Buy when price significantly below VWAP and turning up
                    buy_condition = (df['vwap_deviation'] < -threshold) & (df['close'] > df['close'].shift(1))
                    sell_condition = (df['vwap_deviation'] > threshold) | (df['close'] < df['vwap'])
                else:
                    buy_condition = pd.Series([False] * len(df), index=df.index)
                    sell_condition = pd.Series([False] * len(df), index=df.index)
                
            elif 'adx' in strategy_key:
                if 'adx' in df.columns:
                    adx_threshold = strategy_config['adx_threshold']
                    # Strong trend with ADX
                    buy_condition = (df['adx'] > adx_threshold) & (df['plus_di'] > df['minus_di']) & (df['close'] > df['ema_fast'])
                    sell_condition = (df['adx'] < adx_threshold) | (df['plus_di'] < df['minus_di'])
                else:
                    buy_condition = pd.Series([False] * len(df), index=df.index)
                    sell_condition = pd.Series([False] * len(df), index=df.index)
                
            elif 'support_resistance' in strategy_key:
                # Support/Resistance breakout
                buy_condition = (df['sr_position'] < 0.3) & (df['close'] > df['close'].shift(1))  # Near support and bouncing
                sell_condition = (df['sr_position'] > 0.8) | (df['close'] < df['support'] * 1.01)  # Near resistance or below support
            
            else:
                return df
            
            # Apply signals
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
            return df
            
        except Exception as e:
            print(f"       ❌ Signal generation failed for {strategy_key}: {e}")
            return df
    
    def _backtest_comprehensive_strategy(self, df, strategy_name, strategy_config):
        """Comprehensive backtesting with detailed trade analysis"""
        try:
            trades = []
            portfolio_value = INITIAL_CAPITAL
            position_size = 0
            entry_price = 0
            entry_time = None
            highest_price_since_entry = 0
            
            # Strategy parameters
            profit_target = strategy_config['profit_target']
            stop_loss = strategy_config['stop_loss']
            max_hold_minutes = strategy_config['max_hold_minutes']
            min_hold_minutes = strategy_config.get('min_hold_minutes', 15)
            trailing_trigger = strategy_config.get('trailing_trigger', profit_target * 0.5)
            trailing_distance = strategy_config.get('trailing_distance', stop_loss * 0.5)
            
            # Tracking variables
            trailing_stop_price = 0
            trailing_active = False
            
            for i in range(1, len(df)):
                current_time = pd.to_datetime(df.loc[i, 'datetime' if 'datetime' in df.columns else 'date'])
                current_price = df.loc[i, 'close']
                signal = df.loc[i, 'signal']
                
                # Exit Logic
                if position_size > 0 and entry_price > 0:
                    pct_change = (current_price - entry_price) / entry_price
                    minutes_held = (current_time - entry_time).total_seconds() / 60
                    
                    # Update highest price
                    if current_price > highest_price_since_entry:
                        highest_price_since_entry = current_price
                    
                    # Trailing stop logic
                    if not trailing_active and pct_change >= trailing_trigger:
                        trailing_active = True
                        trailing_stop_price = current_price - (current_price * trailing_distance)
                    
                    if trailing_active:
                        new_trailing_stop = current_price - (current_price * trailing_distance)
                        if new_trailing_stop > trailing_stop_price:
                            trailing_stop_price = new_trailing_stop
                    
                    should_exit = False
                    exit_reason = ""
                    
                    # Exit conditions
                    if minutes_held < min_hold_minutes:
                        continue  # Don't exit before minimum hold
                    elif minutes_held >= max_hold_minutes:
                        should_exit = True
                        exit_reason = "Time Limit"
                    elif trailing_active and current_price <= trailing_stop_price:
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
                        # Calculate trade result
                        exit_value = position_size * current_price * (1 - TRANSACTION_COST)
                        entry_cost = POSITION_SIZE_PER_TRADE * (1 + TRANSACTION_COST)
                        trade_pnl = exit_value - entry_cost
                        portfolio_value += trade_pnl
                        
                        # Detailed trade record
                        max_profit_reached = (highest_price_since_entry - entry_price) / entry_price
                        max_drawdown_in_trade = (entry_price - min(df.loc[entry_time:current_time, 'close'])) / entry_price if len(df.loc[entry_time:current_time, 'close']) > 0 else 0
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position_size': position_size,
                            'pnl': trade_pnl,
                            'return_pct': pct_change * 100,
                            'minutes_held': minutes_held,
                            'exit_reason': exit_reason,
                            'max_profit_reached_pct': max_profit_reached * 100,
                            'max_drawdown_in_trade_pct': max_drawdown_in_trade * 100,
                            'trailing_activated': trailing_active,
                            'hit_stop_loss': exit_reason == "Stop Loss",
                            'hit_profit_target': exit_reason == "Profit Target",
                            'portfolio_value': portfolio_value
                        })
                        
                        # Reset position
                        position_size = 0
                        entry_price = 0
                        entry_time = None
                        trailing_active = False
                        trailing_stop_price = 0
                        highest_price_since_entry = 0
                
                # Entry Logic
                elif (position_size == 0 and 
                      signal == 1 and 
                      portfolio_value >= POSITION_SIZE_PER_TRADE):
                    
                    entry_cost = POSITION_SIZE_PER_TRADE * (1 + TRANSACTION_COST)
                    position_size = POSITION_SIZE_PER_TRADE / current_price
                    entry_price = current_price
                    entry_time = current_time
                    highest_price_since_entry = current_price
                    portfolio_value -= entry_cost
            
            return trades, portfolio_value
            
        except Exception as e:
            print(f"       ❌ Backtest failed: {e}")
            return [], INITIAL_CAPITAL
    
    def _analyze_detailed_performance(self, trades, final_portfolio, strategy_name, timeframe):
        """Detailed performance analysis with stop loss analysis"""
        if not trades:
            return None
        
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = winning_trades / total_trades
        
        # Performance metrics
        total_return = (final_portfolio - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        avg_return_per_trade = df_trades['return_pct'].mean()
        avg_winning_trade = df_trades[df_trades['pnl'] > 0]['return_pct'].mean() if winning_trades > 0 else 0
        avg_losing_trade = df_trades[df_trades['pnl'] < 0]['return_pct'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        max_return = df_trades['return_pct'].max()
        min_return = df_trades['return_pct'].min()
        std_return = df_trades['return_pct'].std()
        sharpe_ratio = avg_return_per_trade / std_return if std_return > 0 else 0
        
        # Exit reason analysis
        exit_reasons = df_trades['exit_reason'].value_counts()
        stop_loss_count = exit_reasons.get('Stop Loss', 0)
        profit_target_count = exit_reasons.get('Profit Target', 0)
        trailing_stop_count = exit_reasons.get('Trailing Stop', 0)
        signal_exit_count = exit_reasons.get('Signal Exit', 0)
        time_limit_count = exit_reasons.get('Time Limit', 0)
        
        # Stop loss analysis
        stop_loss_rate = stop_loss_count / total_trades
        profit_target_rate = profit_target_count / total_trades
        
        # Hold time analysis
        avg_hold_time = df_trades['minutes_held'].mean()
        avg_winning_hold_time = df_trades[df_trades['pnl'] > 0]['minutes_held'].mean() if winning_trades > 0 else 0
        avg_losing_hold_time = df_trades[df_trades['pnl'] < 0]['minutes_held'].mean() if losing_trades > 0 else 0
        
        # Max profit reached analysis
        avg_max_profit_reached = df_trades['max_profit_reached_pct'].mean()
        trades_hit_target = len(df_trades[df_trades['max_profit_reached_pct'] >= (df_trades['return_pct'] * 0.9)])  # Close to target
        
        # Trailing stop effectiveness
        trailing_trades = df_trades[df_trades['trailing_activated'] == True]
        trailing_success_rate = len(trailing_trades[trailing_trades['pnl'] > 0]) / len(trailing_trades) if len(trailing_trades) > 0 else 0
        
        # Consecutive analysis
        consecutive_losses = self._calculate_max_consecutive_losses(df_trades)
        consecutive_wins = self._calculate_max_consecutive_wins(df_trades)
        
        return {
            'strategy_name': strategy_name,
            'timeframe': timeframe,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_portfolio': final_portfolio,
            
            # Return analysis
            'avg_return_per_trade': avg_return_per_trade,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'max_return': max_return,
            'min_return': min_return,
            'sharpe_ratio': sharpe_ratio,
            
            # Exit analysis
            'stop_loss_count': stop_loss_count,
            'stop_loss_rate': stop_loss_rate,
            'profit_target_count': profit_target_count,
            'profit_target_rate': profit_target_rate,
            'trailing_stop_count': trailing_stop_count,
            'signal_exit_count': signal_exit_count,
            'time_limit_count': time_limit_count,
            
            # Hold time analysis
            'avg_hold_time_minutes': avg_hold_time,
            'avg_winning_hold_time': avg_winning_hold_time,
            'avg_losing_hold_time': avg_losing_hold_time,
            
            # Performance analysis
            'avg_max_profit_reached': avg_max_profit_reached,
            'trailing_trades': len(trailing_trades),
            'trailing_success_rate': trailing_success_rate,
            
            # Risk analysis
            'max_consecutive_losses': consecutive_losses,
            'max_consecutive_wins': consecutive_wins,
            
            'status': 'COMPLETED'
        }
    
    def _calculate_max_consecutive_losses(self, df_trades):
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for _, trade in df_trades.iterrows():
            if trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_consecutive_wins(self, df_trades):
        """Calculate maximum consecutive wins"""
        max_consecutive = 0
        current_consecutive = 0
        
        for _, trade in df_trades.iterrows():
            if trade['pnl'] > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis across all indicators and timeframes"""
        print(f"\n🔍 COMPREHENSIVE INTRADAY INDICATOR ANALYSIS")
        print(f"{'='*70}")
        print(f"📊 Testing 14 strategies across 2 timeframes")
        print(f"💰 Initial Capital: ₹{INITIAL_CAPITAL:,}")
        print(f"💳 Position Size: ₹{POSITION_SIZE_PER_TRADE:,}")
        print(f"{'='*70}")
        
        all_results = {}
        
        # Process each timeframe
        for timeframe_key, timeframe_config in COMPREHENSIVE_STRATEGIES.items():
            print(f"\n📈 PROCESSING {timeframe_key.upper()} TIMEFRAME")
            print(f"{'='*50}")
            
            data_dir = timeframe_config['data_dir']
            
            # Get data files
            try:
                data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                print(f"📁 Found {len(data_files)} data files")
            except:
                print(f"❌ Data directory not found: {data_dir}")
                continue
            
            timeframe_results = {}
            
            # Test with first 10 symbols for speed
            test_files = data_files[:10]
            
            # Process each strategy
            for strategy_key, strategy_config in timeframe_config['strategies'].items():
                print(f"\n🎯 STRATEGY: {strategy_config['name']}")
                
                all_trades = []
                
                for i, filename in enumerate(test_files, 1):
                    try:
                        symbol = filename.replace('.csv', '')
                        print(f"   📊 [{i}/{len(test_files)}] {symbol}")
                        
                        # Load data
                        file_path = os.path.join(data_dir, filename)
                        df = pd.read_csv(file_path)
                        
                        if len(df) < 100:
                            print(f"       ⚠️ Insufficient data ({len(df)} bars)")
                            continue
                        
                        # Prepare datetime
                        if 'datetime' not in df.columns and 'date' in df.columns:
                            df['datetime'] = pd.to_datetime(df['date'])
                        
                        # Calculate comprehensive indicators
                        df = self._calculate_comprehensive_indicators(df, timeframe_key)
                        
                        # Generate strategy signals
                        df = self._generate_strategy_signals(df, strategy_key, strategy_config)
                        
                        # Run backtest
                        symbol_trades, symbol_portfolio = self._backtest_comprehensive_strategy(
                            df, strategy_config['name'], strategy_config
                        )
                        
                        if symbol_trades:
                            # Add symbol info
                            for trade in symbol_trades:
                                trade['symbol'] = symbol
                            all_trades.extend(symbol_trades)
                            print(f"       ✅ {len(symbol_trades)} trades")
                        else:
                            print(f"       ⚠️ No trades generated")
                        
                        del df
                        gc.collect()
                        
                    except Exception as e:
                        print(f"       ❌ Error: {e}")
                        continue
                
                # Analyze strategy performance
                if all_trades:
                    final_portfolio = INITIAL_CAPITAL + sum([trade['pnl'] for trade in all_trades])
                    
                    performance = self._analyze_detailed_performance(
                        all_trades, final_portfolio, strategy_config['name'], timeframe_key
                    )
                    
                    if performance:
                        timeframe_results[strategy_key] = performance
                        self.execution_stats['successful_strategies'] += 1
                        self.execution_stats['total_trades_analyzed'] += len(all_trades)
                        
                        print(f"   ✅ Analysis Complete:")
                        print(f"       📊 Trades: {performance['total_trades']}")
                        print(f"       🎯 Win Rate: {performance['win_rate']:.2%}")
                        print(f"       📈 Return: {performance['total_return']:.2f}%")
                        print(f"       🛑 Stop Loss Rate: {performance['stop_loss_rate']:.2%}")
                        print(f"       ⏱️ Avg Hold: {performance['avg_hold_time_minutes']:.1f} min")
                    else:
                        print(f"   ❌ Analysis failed")
                        self.execution_stats['failed_strategies'] += 1
                else:
                    print(f"   ❌ No trades for analysis")
                    self.execution_stats['failed_strategies'] += 1
                
                self.execution_stats['total_strategies_tested'] += 1
            
            all_results[timeframe_key] = timeframe_results
        
        self.results_summary = all_results
        self._save_comprehensive_results()
        self._print_comprehensive_comparison()
        
        return all_results
    
    def _save_comprehensive_results(self):
        """Save comprehensive results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results_file = os.path.join(DATA_DIRS['results'], f'comprehensive_indicator_analysis_{timestamp}.json')
            
            with open(results_file, 'w') as f:
                json.dump(self.results_summary, f, indent=2, default=str)
            
            print(f"📁 Comprehensive results saved: {results_file}")
            
        except Exception as e:
            print(f"⚠️ Results saving failed: {e}")
    
    def _print_comprehensive_comparison(self):
        """Print comprehensive comparison results"""
        end_time = datetime.now()
        duration = end_time - self.execution_stats['start_time']
        
        print(f"\n{'='*80}")
        print(f"🏆 COMPREHENSIVE INTRADAY INDICATOR COMPARISON")
        print(f"{'='*80}")
        
        # Execution stats
        print(f"⏱️ Execution Time: {duration.total_seconds():.1f} seconds")
        print(f"📊 Strategies Tested: {self.execution_stats['total_strategies_tested']}")
        print(f"✅ Successful: {self.execution_stats['successful_strategies']}")
        print(f"❌ Failed: {self.execution_stats['failed_strategies']}")
        print(f"📈 Total Trades: {self.execution_stats['total_trades_analyzed']}")
        
        # Collect all strategies for comparison
        all_strategies = []
        for timeframe_key, strategies in self.results_summary.items():
            for strategy_key, performance in strategies.items():
                if performance and performance.get('status') == 'COMPLETED':
                    all_strategies.append(performance)
        
        if all_strategies:
            # Sort by total return
            all_strategies.sort(key=lambda x: x['total_return'], reverse=True)
            
            # 15-minute strategies
            minute_15_strategies = [s for s in all_strategies if s['timeframe'] == '15minute']
            hour_1_strategies = [s for s in all_strategies if s['timeframe'] == '1hour']
            
            print(f"\n🚀 TOP 15-MINUTE STRATEGIES:")
            print(f"{'='*60}")
            if minute_15_strategies:
                for i, strategy in enumerate(minute_15_strategies[:5], 1):
                    print(f"{i}. {strategy['strategy_name']}")
                    print(f"   📈 Return: {strategy['total_return']:.2f}%")
                    print(f"   🎯 Win Rate: {strategy['win_rate']:.2%}")
                    print(f"   📊 Trades: {strategy['total_trades']}")
                    print(f"   🛑 Stop Loss Rate: {strategy['stop_loss_rate']:.2%}")
                    print(f"   ⏱️ Avg Hold: {strategy['avg_hold_time_minutes']:.1f} min")
                    print(f"   📊 Avg Trade: {strategy['avg_return_per_trade']:.2f}%")
                    print()
            
            print(f"🚀 TOP 1-HOUR STRATEGIES:")
            print(f"{'='*60}")
            if hour_1_strategies:
                for i, strategy in enumerate(hour_1_strategies[:5], 1):
                    print(f"{i}. {strategy['strategy_name']}")
                    print(f"   📈 Return: {strategy['total_return']:.2f}%")
                    print(f"   🎯 Win Rate: {strategy['win_rate']:.2%}")
                    print(f"   📊 Trades: {strategy['total_trades']}")
                    print(f"   🛑 Stop Loss Rate: {strategy['stop_loss_rate']:.2%}")
                    print(f"   ⏱️ Avg Hold: {strategy['avg_hold_time_minutes']:.1f} min")
                    print(f"   📊 Avg Trade: {strategy['avg_return_per_trade']:.2f}%")
                    print()
            
            # Overall best
            print(f"🏆 OVERALL BEST STRATEGIES:")
            print(f"{'='*50}")
            for i, strategy in enumerate(all_strategies[:3], 1):
                print(f"{i}. {strategy['strategy_name']} ({strategy['timeframe']})")
                print(f"   📈 Total Return: {strategy['total_return']:.2f}%")
                print(f"   💰 Final Portfolio: ₹{strategy['final_portfolio']:,.0f}")
                print(f"   🎯 Win Rate: {strategy['win_rate']:.2%}")
                print(f"   📊 Total Trades: {strategy['total_trades']}")
                print(f"   📊 Avg Return/Trade: {strategy['avg_return_per_trade']:.2f}%")
                print(f"   🛑 Stop Loss Hits: {strategy['stop_loss_count']} ({strategy['stop_loss_rate']:.1%})")
                print(f"   🎯 Profit Target Hits: {strategy['profit_target_count']} ({strategy['profit_target_rate']:.1%})")
                print(f"   ⏱️ Avg Hold Time: {strategy['avg_hold_time_minutes']:.1f} minutes")
                print(f"   📊 Sharpe Ratio: {strategy['sharpe_ratio']:.2f}")
                print()
            
            # Stop loss analysis summary
            print(f"🛑 STOP LOSS ANALYSIS SUMMARY:")
            print(f"{'='*40}")
            for strategy in all_strategies[:5]:
                print(f"{strategy['strategy_name']} ({strategy['timeframe']})")
                print(f"   Stop Loss Hits: {strategy['stop_loss_count']}/{strategy['total_trades']} ({strategy['stop_loss_rate']:.1%})")
                print(f"   Avg Losing Trade: {strategy['avg_losing_trade']:.2f}%")
                print(f"   Max Consecutive Losses: {strategy['max_consecutive_losses']}")
                print()

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE INTRADAY INDICATOR COMPARISON")
    print("="*60)
    print("📊 Testing 14 different technical indicators")
    print("⏱️ Both 15-minute and 1-hour timeframes")
    print("🛑 Detailed stop loss and performance analysis")
    print("📈 Average trade performance tracking")
    print("="*60)
    
    try:
        analyzer = ComprehensiveIntradayAnalyzer()
        results = analyzer.run_comprehensive_analysis()
        
        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print(f"🏆 Best indicators identified for each timeframe")
        print(f"🛑 Stop loss analysis complete")
        print(f"📊 Ready for optimal strategy selection")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise
