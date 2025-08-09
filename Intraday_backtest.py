# comprehensive_cnc_strategy_comparison.py

import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
from datetime import datetime, timedelta
import gc

warnings.filterwarnings("ignore")

# ========================
# COMPREHENSIVE CNC STRATEGY COMPARISON
# ========================
BASE_DIR = "/root/falah-ai-bot/"
DATA_DIRS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    'results': os.path.join(BASE_DIR, "cnc_strategy_comparison"),
}

os.makedirs(DATA_DIRS['results'], exist_ok=True)

# COMPREHENSIVE CNC STRATEGIES (ALL HELD MINIMUM 1 DAY)
CNC_STRATEGY_COMPARISON = {
    'daily_strategies': {
        'data_dir': DATA_DIRS['daily'],
        'timeframe': 'daily',
        'strategies': {
            'williams_r_cnc': {
                'name': 'Williams %R CNC (Daily)',
                'profit_target': 0.12,          # 12%
                'stop_loss': 0.05,              # 5%
                'oversold_threshold': -80,
                'overbought_threshold': -20,
                'min_hold_days': 1,
                'max_hold_days': 20,
                'trailing_trigger': 0.06,
                'trailing_distance': 0.025
            },
            'rsi_cnc': {
                'name': 'RSI CNC (Daily)',
                'profit_target': 0.10,          # 10%
                'stop_loss': 0.05,              # 5%
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'min_hold_days': 1,
                'max_hold_days': 15,
                'trailing_trigger': 0.05,
                'trailing_distance': 0.025
            },
            'macd_cnc': {
                'name': 'MACD CNC (Daily)',
                'profit_target': 0.15,          # 15%
                'stop_loss': 0.06,              # 6%
                'min_hold_days': 1,
                'max_hold_days': 25,
                'trailing_trigger': 0.08,
                'trailing_distance': 0.03
            },
            'bollinger_cnc': {
                'name': 'Bollinger Bands CNC (Daily)',
                'profit_target': 0.12,          # 12%
                'stop_loss': 0.05,              # 5%
                'min_hold_days': 1,
                'max_hold_days': 18,
                'trailing_trigger': 0.06,
                'trailing_distance': 0.025
            },
            'stochastic_cnc': {
                'name': 'Stochastic CNC (Daily)',
                'profit_target': 0.11,          # 11%
                'stop_loss': 0.05,              # 5%
                'stoch_oversold': 20,
                'stoch_overbought': 80,
                'min_hold_days': 1,
                'max_hold_days': 16,
                'trailing_trigger': 0.055,
                'trailing_distance': 0.025
            },
            'ema_crossover_cnc': {
                'name': 'EMA Crossover CNC (Daily)',
                'profit_target': 0.10,          # 10%
                'stop_loss': 0.04,              # 4%
                'ema_fast': 10,
                'ema_slow': 21,
                'min_hold_days': 1,
                'max_hold_days': 15,
                'trailing_trigger': 0.05,
                'trailing_distance': 0.02
            },
            'adx_trend_cnc': {
                'name': 'ADX Trend CNC (Daily)',
                'profit_target': 0.18,          # 18%
                'stop_loss': 0.07,              # 7%
                'adx_threshold': 25,
                'min_hold_days': 1,
                'max_hold_days': 30,
                'trailing_trigger': 0.10,
                'trailing_distance': 0.035
            },
            'support_resistance_cnc': {
                'name': 'Support/Resistance CNC (Daily)',
                'profit_target': 0.14,          # 14%
                'stop_loss': 0.06,              # 6%
                'lookback_period': 20,
                'min_hold_days': 1,
                'max_hold_days': 22,
                'trailing_trigger': 0.08,
                'trailing_distance': 0.03
            }
        }
    },
    'timing_strategies': {
        'data_dir': DATA_DIRS['15minute'],
        'timeframe': '15minute',
        'strategies': {
            'precision_williams_cnc': {
                'name': 'Precision Williams %R CNC (15min timing)',
                'profit_target': 0.08,          # 8%
                'stop_loss': 0.04,              # 4%
                'oversold_threshold': -85,
                'overbought_threshold': -15,
                'min_hold_days': 1,              # CNC REQUIREMENT
                'max_hold_days': 5,
                'trailing_trigger': 0.04,
                'trailing_distance': 0.02
            },
            'precision_rsi_cnc': {
                'name': 'Precision RSI CNC (15min timing)',
                'profit_target': 0.06,          # 6%
                'stop_loss': 0.03,              # 3%
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'min_hold_days': 1,              # CNC REQUIREMENT
                'max_hold_days': 4,
                'trailing_trigger': 0.03,
                'trailing_distance': 0.015
            },
            'precision_macd_cnc': {
                'name': 'Precision MACD CNC (15min timing)',
                'profit_target': 0.10,          # 10%
                'stop_loss': 0.05,              # 5%
                'min_hold_days': 1,              # CNC REQUIREMENT
                'max_hold_days': 7,
                'trailing_trigger': 0.05,
                'trailing_distance': 0.025
            }
        }
    }
}

# ZERODHA CNC CHARGES (Same as before)
ZERODHA_CNC_CHARGES = {
    'brokerage_rate': 0.0000,           # ₹0 for CNC
    'stt_buy_rate': 0.001,              # 0.1% on buy
    'stt_sell_rate': 0.001,             # 0.1% on sell
    'exchange_txn_rate': 0.0000345,     # 0.00345%
    'gst_rate': 0.18,                   # 18% GST
    'sebi_rate': 0.000001,              # ₹1 per crore
    'stamp_duty_rate': 0.00015,         # 0.015% on buy
    'dp_charges_per_sell': 13.5         # ₹13.5 per sell
}

# TRADING PARAMETERS
INITIAL_CAPITAL = 1000000
POSITION_SIZE_PER_TRADE = 100000

class ComprehensiveCNCComparison:
    def __init__(self):
        self.results_summary = {}
        self.execution_stats = {
            'total_strategies_tested': 0,
            'successful_strategies': 0,
            'failed_strategies': 0,
            'total_trades_analyzed': 0,
            'total_cnc_charges': 0,
            'start_time': datetime.now()
        }
        
        print("🕌 Comprehensive CNC Strategy Comparison Initialized")
        print("✅ ALL strategies will be CNC (minimum 1-day holding)")
        print("✅ Fair comparison using identical parameters")
    
    def _calculate_cnc_charges(self, buy_value, sell_value):
        """Calculate comprehensive CNC charges"""
        charges = ZERODHA_CNC_CHARGES
        
        # All charge components
        brokerage = 0  # Free for CNC
        stt = (buy_value * charges['stt_buy_rate']) + (sell_value * charges['stt_sell_rate'])
        exchange = (buy_value + sell_value) * charges['exchange_txn_rate']
        gst = exchange * charges['gst_rate']
        sebi = (buy_value + sell_value) * charges['sebi_rate']
        stamp_duty = buy_value * charges['stamp_duty_rate']
        dp_charges = charges['dp_charges_per_sell']
        
        total_charges = brokerage + stt + exchange + gst + sebi + stamp_duty + dp_charges
        
        return {
            'total_charges': total_charges,
            'stt': stt,
            'exchange_charges': exchange,
            'stamp_duty': stamp_duty,
            'dp_charges': dp_charges,
            'effective_cost_percentage': total_charges / buy_value * 100
        }
    
    def _calculate_comprehensive_indicators(self, df, timeframe):
        """Calculate all technical indicators"""
        try:
            print(f"       📊 Calculating indicators for {timeframe}...")
            
            # Williams %R
            if 'williams_r' not in df.columns:
                highest_high = df['high'].rolling(14).max()
                lowest_low = df['low'].rolling(14).min()
                df['williams_r'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
            
            # RSI
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
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic
            lowest_low_14 = df['low'].rolling(14).min()
            highest_high_14 = df['high'].rolling(14).max()
            df['stoch_k'] = ((df['close'] - lowest_low_14) / (highest_high_14 - lowest_low_14)) * 100
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # EMA
            if timeframe == 'daily':
                df['ema_fast'] = df['close'].ewm(span=10).mean()
                df['ema_slow'] = df['close'].ewm(span=21).mean()
            else:  # 15minute for timing
                df['ema_fast'] = df['close'].ewm(span=5).mean()
                df['ema_slow'] = df['close'].ewm(span=13).mean()
            
            # ADX (for daily only)
            if timeframe == 'daily':
                # Simplified ADX
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
            
            # Support and Resistance
            lookback = 20
            df['resistance'] = df['high'].rolling(lookback).max()
            df['support'] = df['low'].rolling(lookback).min()
            df['sr_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
            
            print(f"       ✅ All indicators calculated")
            return df
            
        except Exception as e:
            print(f"       ❌ Indicator calculation failed: {e}")
            return df
    
    def _generate_cnc_signals(self, df, strategy_key, strategy_config):
        """Generate trading signals for CNC strategies"""
        try:
            df['signal'] = 0
            
            if 'williams' in strategy_key:
                oversold = strategy_config['oversold_threshold']
                overbought = strategy_config['overbought_threshold']
                buy_condition = (df['williams_r'] < oversold) & (df['williams_r'] > df['williams_r'].shift(1))
                sell_condition = df['williams_r'] > overbought
                
            elif 'rsi' in strategy_key:
                oversold = strategy_config['rsi_oversold']
                overbought = strategy_config['rsi_overbought']
                buy_condition = (df['rsi'] < oversold) & (df['rsi'] > df['rsi'].shift(1))
                sell_condition = df['rsi'] > overbought
                
            elif 'macd' in strategy_key:
                buy_condition = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
                sell_condition = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
                
            elif 'bollinger' in strategy_key:
                buy_condition = (df['bb_position'] < 0.2) & (df['close'] > df['close'].shift(1))
                sell_condition = (df['bb_position'] > 0.8) | (df['close'] < df['bb_middle'])
                
            elif 'stochastic' in strategy_key:
                oversold = strategy_config['stoch_oversold']
                overbought = strategy_config['stoch_overbought']
                buy_condition = (df['stoch_k'] < oversold) & (df['stoch_k'] > df['stoch_d'])
                sell_condition = df['stoch_k'] > overbought
                
            elif 'ema' in strategy_key:
                buy_condition = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
                sell_condition = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
                
            elif 'adx' in strategy_key:
                if 'adx' in df.columns:
                    adx_threshold = strategy_config['adx_threshold']
                    buy_condition = (df['adx'] > adx_threshold) & (df['plus_di'] > df['minus_di']) & (df['close'] > df['ema_fast'])
                    sell_condition = (df['adx'] < adx_threshold) | (df['plus_di'] < df['minus_di'])
                else:
                    buy_condition = pd.Series([False] * len(df), index=df.index)
                    sell_condition = pd.Series([False] * len(df), index=df.index)
                
            elif 'support_resistance' in strategy_key:
                buy_condition = (df['sr_position'] < 0.3) & (df['close'] > df['close'].shift(1))
                sell_condition = (df['sr_position'] > 0.8) | (df['close'] < df['support'] * 1.01)
            
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
            print(f"       ❌ Signal generation failed: {e}")
            df['signal'] = 0
            return df
    
    def _backtest_cnc_strategy(self, df, strategy_name, strategy_config):
        """CNC backtesting with minimum 1-day holding"""
        try:
            trades = []
            portfolio_value = INITIAL_CAPITAL
            active_positions = {}
            position_id = 0
            
            # Strategy parameters
            profit_target = strategy_config['profit_target']
            stop_loss = strategy_config['stop_loss']
            min_hold_days = strategy_config.get('min_hold_days', 1)  # CNC minimum
            max_hold_days = strategy_config['max_hold_days']
            trailing_trigger = strategy_config.get('trailing_trigger', profit_target * 0.5)
            trailing_distance = strategy_config.get('trailing_distance', stop_loss * 0.5)
            max_positions = 10  # Limit concurrent positions
            
            for i in range(1, len(df)):
                current_date = pd.to_datetime(df.iloc[i]['date' if 'date' in df.columns else 'datetime'])
                current_price = df.iloc[i]['close']
                signal = df.iloc[i]['signal']
                
                if not np.isfinite(current_price) or current_price <= 0:
                    continue
                
                # Check existing positions for exits
                positions_to_close = []
                
                for pos_id, position in active_positions.items():
                    entry_date = position['entry_date']
                    entry_price = position['entry_price']
                    days_held = (current_date - entry_date).days
                    pct_change = (current_price - entry_price) / entry_price
                    
                    # Update highest price for trailing
                    if current_price > position['highest_price']:
                        position['highest_price'] = current_price
                    
                    # Update trailing stop
                    if not position['trailing_active'] and pct_change >= trailing_trigger:
                        position['trailing_active'] = True
                        position['trailing_stop'] = current_price * (1 - trailing_distance)
                    
                    if position['trailing_active']:
                        new_trailing_stop = current_price * (1 - trailing_distance)
                        if new_trailing_stop > position['trailing_stop']:
                            position['trailing_stop'] = new_trailing_stop
                    
                    should_exit = False
                    exit_reason = ""
                    
                    # CNC EXIT CONDITIONS (minimum 1 day holding)
                    if days_held < min_hold_days:
                        continue  # Cannot exit before 1 day (CNC requirement)
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
                        # Calculate CNC charges
                        buy_value = POSITION_SIZE_PER_TRADE
                        sell_value = position['shares'] * current_price
                        charge_details = self._calculate_cnc_charges(buy_value, sell_value)
                        total_charges = charge_details['total_charges']
                        
                        # Net profit/loss
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
                        self.execution_stats['total_cnc_charges'] += total_charges
                
                # Close marked positions
                for pos_id in positions_to_close:
                    del active_positions[pos_id]
                
                # Entry Logic - CNC ONLY
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
            
            # Close remaining positions at end (if held minimum 1 day)
            final_date = pd.to_datetime(df.iloc[len(df)-1]['date' if 'date' in df.columns else 'datetime'])
            final_price = df.iloc[len(df)-1]['close']
            
            for pos_id, position in active_positions.items():
                days_held = (final_date - position['entry_date']).days
                if days_held >= min_hold_days:
                    pct_change = (final_price - position['entry_price']) / position['entry_price']
                    
                    buy_value = POSITION_SIZE_PER_TRADE
                    sell_value = position['shares'] * final_price
                    charge_details = self._calculate_cnc_charges(buy_value, sell_value)
                    total_charges = charge_details['total_charges']
                    
                    gross_pnl = sell_value - buy_value
                    net_pnl = gross_pnl - total_charges
                    portfolio_value += net_pnl
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': final_date,
                        'entry_price': position['entry_price'],
                        'exit_price': final_price,
                        'shares': position['shares'],
                        'days_held': days_held,
                        'gross_pnl': gross_pnl,
                        'net_pnl': net_pnl,
                        'total_charges': total_charges,
                        'return_pct': pct_change * 100,
                        'exit_reason': "End of Period",
                        'trailing_activated': position['trailing_active'],
                        'portfolio_value': portfolio_value
                    })
            
            return trades, portfolio_value
            
        except Exception as e:
            print(f"       ❌ CNC backtest failed: {e}")
            return [], INITIAL_CAPITAL
    
    def _analyze_cnc_performance(self, trades, final_portfolio, strategy_name):
        """Analyze CNC performance"""
        if not trades:
            return None
        
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['net_pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_return = (final_portfolio - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        avg_return_per_trade = df_trades['return_pct'].mean()
        avg_hold_days = df_trades['days_held'].mean()
        
        # Charge analysis
        total_charges = df_trades['total_charges'].sum()
        avg_charges_per_trade = total_charges / total_trades if total_trades > 0 else 0
        
        # Net vs Gross
        total_gross_pnl = df_trades['gross_pnl'].sum()
        total_net_pnl = df_trades['net_pnl'].sum()
        charges_impact = (total_gross_pnl - total_net_pnl) / abs(total_gross_pnl) * 100 if total_gross_pnl != 0 else 0
        
        # Risk metrics
        std_return = df_trades['return_pct'].std()
        sharpe_ratio = avg_return_per_trade / std_return if std_return > 0 else 0
        
        # Max drawdown
        portfolio_values = df_trades['portfolio_value'].values
        max_drawdown = 0
        if len(portfolio_values) > 0:
            peak = INITIAL_CAPITAL
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        
        # Exit analysis
        exit_reasons = df_trades['exit_reason'].value_counts()
        
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
            'total_gross_pnl': total_gross_pnl,
            'total_net_pnl': total_net_pnl,
            'exit_reasons': exit_reasons.to_dict(),
            'status': 'COMPLETED'
        }
    
    def run_comprehensive_cnc_comparison(self):
        """Run comprehensive CNC comparison"""
        print(f"\n🕌 COMPREHENSIVE CNC STRATEGY COMPARISON")
        print(f"{'='*70}")
        print(f"✅ ALL strategies tested with CNC (minimum 1-day holding)")
        print(f"💰 Initial Capital: ₹{INITIAL_CAPITAL:,}")
        print(f"💳 Position Size: ₹{POSITION_SIZE_PER_TRADE:,}")
        print(f"{'='*70}")
        
        all_results = {}
        
        # Process each strategy group
        for group_key, group_config in CNC_STRATEGY_COMPARISON.items():
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
            test_files = data_files[:20]  # Test 20 symbols for comprehensive results
            
            for strategy_key, strategy_config in group_config['strategies'].items():
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
                        
                        # Calculate indicators
                        df = self._calculate_comprehensive_indicators(df, timeframe)
                        
                        # Generate CNC signals
                        df = self._generate_cnc_signals(df, strategy_key, strategy_config)
                        
                        # Run CNC backtest
                        symbol_trades, symbol_portfolio = self._backtest_cnc_strategy(
                            df, strategy_config['name'], strategy_config
                        )
                        
                        if symbol_trades:
                            for trade in symbol_trades:
                                trade['symbol'] = symbol
                            all_trades.extend(symbol_trades)
                            print(f"       ✅ {len(symbol_trades)} CNC trades")
                        
                        del df
                        gc.collect()
                        
                        self.execution_stats['total_strategies_tested'] += 1
                        
                    except Exception as e:
                        print(f"       ❌ Error: {e}")
                        continue
                
                # Analyze performance
                if all_trades:
                    final_portfolio = INITIAL_CAPITAL + sum([trade['net_pnl'] for trade in all_trades])
                    
                    performance = self._analyze_cnc_performance(
                        all_trades, final_portfolio, strategy_config['name']
                    )
                    
                    if performance:
                        group_results[strategy_key] = performance
                        self.execution_stats['successful_strategies'] += 1
                        self.execution_stats['total_trades_analyzed'] += len(all_trades)
                        
                        print(f"   ✅ CNC Analysis Complete:")
                        print(f"       📊 Trades: {performance['total_trades']}")
                        print(f"       🎯 Win Rate: {performance['win_rate']:.2%}")
                        print(f"       📈 Total Return: {performance['total_return']:.2f}%")
                        print(f"       📅 Avg Hold: {performance['avg_hold_days']:.1f} days")
                        print(f"       💸 Avg Charges: ₹{performance['avg_charges_per_trade']:.0f}/trade")
                else:
                    print(f"   ❌ No CNC trades generated")
            
            all_results[group_key] = group_results
        
        self.results_summary = all_results
        self._save_results()
        self._print_comprehensive_comparison()
        
        return all_results
    
    def _save_results(self):
        """Save results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(DATA_DIRS['results'], f'cnc_comparison_{timestamp}.json')
            
            with open(results_file, 'w') as f:
                json.dump(self.results_summary, f, indent=2, default=str)
            
            print(f"📁 Results saved: {results_file}")
            
        except Exception as e:
            print(f"⚠️ Save failed: {e}")
    
    def _print_comprehensive_comparison(self):
        """Print comprehensive comparison"""
        end_time = datetime.now()
        duration = end_time - self.execution_stats['start_time']
        
        print(f"\n{'='*80}")
        print(f"🏆 COMPREHENSIVE CNC STRATEGY COMPARISON RESULTS")
        print(f"{'='*80}")
        
        print(f"⏱️ Execution Time: {duration.total_seconds():.1f} seconds")
        print(f"📊 Strategies Tested: {self.execution_stats['total_strategies_tested']}")
        print(f"✅ Successful: {self.execution_stats['successful_strategies']}")
        print(f"📈 Total CNC Trades: {self.execution_stats['total_trades_analyzed']}")
        print(f"💰 Total CNC Charges: ₹{self.execution_stats['total_cnc_charges']:,.0f}")
        
        # Collect all strategies
        all_strategies = []
        for group_key, strategies in self.results_summary.items():
            for strategy_key, performance in strategies.items():
                if performance and performance.get('status') == 'COMPLETED':
                    all_strategies.append(performance)
        
        if all_strategies:
            # Sort by total return
            all_strategies.sort(key=lambda x: x['total_return'], reverse=True)
            
            print(f"\n🏆 TOP CNC STRATEGIES (FAIR COMPARISON):")
            print(f"{'='*70}")
            
            for i, strategy in enumerate(all_strategies, 1):
                print(f"{i}. {strategy['strategy_name']}")
                print(f"   📈 Total Return: {strategy['total_return']:.2f}%")
                print(f"   💰 Final Portfolio: ₹{strategy['final_portfolio']:,.0f}")
                print(f"   🎯 Win Rate: {strategy['win_rate']:.2%}")
                print(f"   📊 Total Trades: {strategy['total_trades']}")
                print(f"   📅 Avg Hold: {strategy['avg_hold_days']:.1f} days")
                print(f"   📊 Avg Return/Trade: {strategy['avg_return_per_trade']:.2f}%")
                print(f"   💸 Avg Charges: ₹{strategy['avg_charges_per_trade']:.0f}/trade")
                print(f"   📉 Charges Impact: {strategy['charges_impact_pct']:.2f}%")
                print(f"   📊 Sharpe Ratio: {strategy['sharpe_ratio']:.2f}")
                print(f"   📉 Max Drawdown: {strategy['max_drawdown']:.2f}%")
                print()
            
            # Performance tiers
            profitable_strategies = [s for s in all_strategies if s['total_return'] > 0]
            breakeven_strategies = [s for s in all_strategies if -2 <= s['total_return'] <= 0]
            losing_strategies = [s for s in all_strategies if s['total_return'] < -2]
            
            print(f"📊 STRATEGY PERFORMANCE TIERS:")
            print(f"✅ Profitable (>0%): {len(profitable_strategies)} strategies")
            print(f"⚖️ Break-even (-2% to 0%): {len(breakeven_strategies)} strategies")
            print(f"❌ Losing (<-2%): {len(losing_strategies)} strategies")
        
        else:
            print("❌ No successful CNC strategies found")
        
        print(f"\n🎯 CNC COMPARISON COMPLETE!")
        print(f"🏆 Williams %R CNC performance can now be fairly compared!")

if __name__ == "__main__":
    print("🕌 COMPREHENSIVE CNC STRATEGY COMPARISON")
    print("="*60)
    print("✅ ALL strategies tested with CNC compliance")
    print("✅ Minimum 1-day holding period")
    print("✅ Complete Zerodha CNC charges")
    print("✅ Fair comparison with identical rules")
    print("="*60)
    
    try:
        comparator = ComprehensiveCNCComparison()
        results = comparator.run_comprehensive_cnc_comparison()
        
        print(f"\n🎉 COMPREHENSIVE CNC COMPARISON COMPLETED!")
        print(f"🏆 Now you can see which strategy truly performs best in CNC mode!")
        print(f"🕌 All results are 100% halal compliant!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
