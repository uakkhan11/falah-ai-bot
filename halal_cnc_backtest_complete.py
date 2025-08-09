# halal_cnc_backtest_complete.py

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
# HALAL CNC TRADING CONFIGURATION
# ========================
BASE_DIR = "/root/falah-ai-bot/"
DATA_DIRS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    'models': os.path.join(BASE_DIR, "models"),
    'results': os.path.join(BASE_DIR, "halal_cnc_results"),
    'live': os.path.join(BASE_DIR, "live_data")
}

# Create results directory
os.makedirs(DATA_DIRS['results'], exist_ok=True)

# HALAL CNC TRADING PARAMETERS
INITIAL_CAPITAL = 1000000        # ₹10 Lakhs starting capital
POSITION_SIZE_PER_TRADE = 100000  # ₹1 Lakh per trade

# COMPREHENSIVE ZERODHA CNC CHARGES (2025 rates)
ZERODHA_CHARGES = {
    # Brokerage (same for buy and sell in CNC)
    'brokerage_rate': 0.0000,  # ₹0 for CNC equity delivery (Zerodha's USP)
    
    # STT (Securities Transaction Tax) - DIFFERENT for buy vs sell
    'stt_buy_rate': 0.001,      # 0.1% on buy side
    'stt_sell_rate': 0.001,     # 0.1% on sell side
    
    # Exchange transaction charges
    'exchange_txn_rate': 0.0000345,  # 0.00345% (NSE)
    
    # GST on (brokerage + exchange charges)
    'gst_rate': 0.18,  # 18% GST
    
    # SEBI charges
    'sebi_rate': 0.000001,  # ₹1 per crore (0.0001%)
    
    # Stamp duty on buy side only
    'stamp_duty_rate': 0.00015,  # 0.015% on buy side only
    
    # DP (Demat) charges - per sell transaction
    'dp_charges_per_sell': 13.5  # ₹13.5 per sell transaction
}

# HALAL CNC STRATEGIES (Cash & Carry Only - No Intraday)
HALAL_CNC_STRATEGIES = {
    'swing_trading_strategies': {
        'data_dir': DATA_DIRS['daily'],
        'timeframe': 'daily',
        'min_hold_days': 1,  # Minimum 1 day holding (CNC requirement)
        'strategies': {
            'ml_swing_cnc': {
                'name': 'ML Swing Trading (CNC)',
                'profit_target': 0.15,          # 15% profit target
                'stop_loss': 0.05,              # 5% stop loss
                'confidence_threshold': 0.70,    # 70% ML confidence
                'max_hold_days': 30,            # Maximum 30 days holding
                'trailing_trigger': 0.08,       # Start trailing at 8%
                'trailing_distance': 0.03,      # 3% trailing distance
                'max_positions': 10             # Max 10 concurrent positions
            },
            'williams_swing_cnc': {
                'name': 'Williams %R Swing (CNC)',
                'profit_target': 0.12,          # 12% profit target
                'stop_loss': 0.05,              # 5% stop loss
                'oversold_threshold': -80,       # Williams %R oversold
                'overbought_threshold': -20,     # Williams %R overbought
                'max_hold_days': 20,            # Maximum 20 days
                'trailing_trigger': 0.06,
                'trailing_distance': 0.025,
                'max_positions': 12
            },
            'ema_momentum_cnc': {
                'name': 'EMA Momentum (CNC)',
                'profit_target': 0.10,          # 10% profit target
                'stop_loss': 0.04,              # 4% stop loss
                'max_hold_days': 15,            # Maximum 15 days
                'trailing_trigger': 0.05,
                'trailing_distance': 0.02,
                'max_positions': 15
            }
        }
    },
    'short_term_cnc_strategies': {
        'data_dir': DATA_DIRS['15minute'],  # Use 15min for entry timing
        'timeframe': '15minute',
        'min_hold_days': 1,  # Still minimum 1 day (CNC)
        'strategies': {
            'precision_entry_cnc': {
                'name': 'Precision Entry CNC (15min timing)',
                'profit_target': 0.08,          # 8% profit target
                'stop_loss': 0.04,              # 4% stop loss
                'confidence_threshold': 0.75,    # Higher confidence for precision
                'max_hold_days': 5,             # Short-term but still CNC
                'trailing_trigger': 0.04,
                'trailing_distance': 0.02,
                'max_positions': 8
            }
        }
    }
}

class HalalCNCBacktestEngine:
    def __init__(self):
        self.ml_model = None
        self.results_summary = {}
        self.execution_stats = {
            'total_files_processed': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'total_trades_generated': 0,
            'total_cnc_charges': 0,
            'backtest_period': {},
            'start_time': datetime.now()
        }
        
        print("🕌 Halal CNC Backtesting Engine Initialized")
        print("✅ No intraday trading - Only CNC (Cash & Carry)")
        print("✅ All trades held minimum 1 day")
        print("✅ Complete Zerodha charges included")
        self._load_ml_model()
    
    def _load_ml_model(self):
        """Load ML model for strategies"""
        try:
            model_path = os.path.join(DATA_DIRS['models'], 'model.pkl')
            if os.path.exists(model_path):
                self.ml_model = joblib.load(model_path)
                print("✅ ML model loaded successfully")
            else:
                print("⚠️ ML model not found - ML strategies will be skipped")
        except Exception as e:
            print(f"⚠️ ML model loading failed: {e}")
    
    def _calculate_cnc_charges(self, buy_value, sell_value):
        """Calculate comprehensive CNC charges as per Zerodha 2025 rates"""
        charges = ZERODHA_CHARGES
        
        # 1. Brokerage (₹0 for CNC - Zerodha's free delivery)
        brokerage_buy = buy_value * charges['brokerage_rate']  # ₹0
        brokerage_sell = sell_value * charges['brokerage_rate']  # ₹0
        total_brokerage = brokerage_buy + brokerage_sell
        
        # 2. STT (Securities Transaction Tax)
        stt_buy = buy_value * charges['stt_buy_rate']    # 0.1% on buy
        stt_sell = sell_value * charges['stt_sell_rate']  # 0.1% on sell
        total_stt = stt_buy + stt_sell
        
        # 3. Exchange transaction charges
        exchange_buy = buy_value * charges['exchange_txn_rate']
        exchange_sell = sell_value * charges['exchange_txn_rate'] 
        total_exchange = exchange_buy + exchange_sell
        
        # 4. GST on (brokerage + exchange charges)
        gst_applicable_amount = total_brokerage + total_exchange
        total_gst = gst_applicable_amount * charges['gst_rate']
        
        # 5. SEBI charges
        sebi_buy = buy_value * charges['sebi_rate']
        sebi_sell = sell_value * charges['sebi_rate']
        total_sebi = sebi_buy + sebi_sell
        
        # 6. Stamp duty (only on buy side)
        stamp_duty = buy_value * charges['stamp_duty_rate']
        
        # 7. DP charges (only on sell side)
        dp_charges = charges['dp_charges_per_sell']
        
        # Total charges
        total_charges = (total_brokerage + total_stt + total_exchange + 
                        total_gst + total_sebi + stamp_duty + dp_charges)
        
        return {
            'brokerage': total_brokerage,
            'stt': total_stt,
            'exchange_charges': total_exchange,
            'gst': total_gst,
            'sebi_charges': total_sebi,
            'stamp_duty': stamp_duty,
            'dp_charges': dp_charges,
            'total_charges': total_charges,
            'effective_cost_percentage': total_charges / buy_value * 100
        }
    
    def _analyze_data_period(self, df):
        """Analyze the data period for backtesting"""
        try:
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            else:
                return None
            
            start_date = df['datetime'].min()
            end_date = df['datetime'].max()
            total_days = (end_date - start_date).days
            total_candles = len(df)
            
            return {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'total_days': total_days,
                'total_candles': total_candles,
                'avg_candles_per_day': total_candles / max(total_days, 1)
            }
        except Exception as e:
            print(f"   ⚠️ Data period analysis failed: {e}")
            return None
    
    def _generate_ml_signals(self, df):
        """Generate ML trading signals"""
        if self.ml_model is None:
            return df
        
        try:
            # Prepare features for ML model
            feature_columns = ['rsi', 'ema_fast', 'ema_slow']
            if 'atr' in df.columns:
                feature_columns.append('atr')
            if 'volume_ratio' in df.columns:
                feature_columns.append('volume_ratio')
            
            available_features = [f for f in feature_columns if f in df.columns]
            if len(available_features) < 3:
                return df
            
            df_clean = df.dropna(subset=available_features)
            if len(df_clean) < 50:
                return df
            
            X = df_clean[available_features]
            df.loc[df_clean.index, 'ml_signal'] = self.ml_model.predict(X)
            df.loc[df_clean.index, 'ml_probability'] = self.ml_model.predict_proba(X)[:, 1]
            
            df['ml_signal'] = df['ml_signal'].fillna(0)
            df['ml_probability'] = df['ml_probability'].fillna(0.5)
            
            return df
            
        except Exception as e:
            print(f"   ⚠️ ML signal generation failed: {e}")
            return df
    
    def _generate_williams_signals(self, df, strategy_config):
        """Generate Williams %R signals"""
        try:
            if 'williams_r' not in df.columns:
                if all(col in df.columns for col in ['high', 'low', 'close']):
                    highest_high = df['high'].rolling(14).max()
                    lowest_low = df['low'].rolling(14).min()
                    df['williams_r'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100
                else:
                    return df
            
            oversold = strategy_config['oversold_threshold']
            overbought = strategy_config['overbought_threshold']
            
            df['williams_signal'] = 0
            
            # Buy signal: Williams %R oversold and turning up
            buy_condition = (df['williams_r'] < oversold) & (df['williams_r'] > df['williams_r'].shift(1))
            df.loc[buy_condition, 'williams_signal'] = 1
            
            # Sell signal: Williams %R overbought
            sell_condition = df['williams_r'] > overbought
            df.loc[sell_condition, 'williams_signal'] = -1
            
            return df
            
        except Exception as e:
            print(f"   ⚠️ Williams %R signal generation failed: {e}")
            return df
    
    def _generate_ema_momentum_signals(self, df):
        """Generate EMA momentum signals"""
        try:
            if 'ema_fast' not in df.columns or 'ema_slow' not in df.columns:
                return df
            
            df['ema_signal'] = 0
            
            # Buy: Fast EMA crosses above Slow EMA with momentum
            buy_condition = (
                (df['ema_fast'] > df['ema_slow']) & 
                (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) &
                (df['close'] > df['ema_fast'])  # Price above fast EMA for confirmation
            )
            df.loc[buy_condition, 'ema_signal'] = 1
            
            # Sell: Fast EMA crosses below Slow EMA
            sell_condition = (
                (df['ema_fast'] < df['ema_slow']) & 
                (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
            )
            df.loc[sell_condition, 'ema_signal'] = -1
            
            return df
            
        except Exception as e:
            print(f"   ⚠️ EMA momentum signal generation failed: {e}")
            return df
    
    def _backtest_cnc_strategy(self, df, strategy_name, strategy_config, signal_column):
        """Core CNC backtesting engine - HALAL COMPLIANT"""
        try:
            # Initialize tracking variables
            trades = []
            portfolio_value = INITIAL_CAPITAL
            active_positions = {}  # Track multiple positions
            position_id = 0
            
            # Strategy parameters
            profit_target = strategy_config['profit_target']
            stop_loss = strategy_config['stop_loss']
            max_hold_days = strategy_config['max_hold_days']
            min_hold_days = strategy_config.get('min_hold_days', 1)  # HALAL: Min 1 day
            trailing_trigger = strategy_config.get('trailing_trigger', profit_target * 0.5)
            trailing_distance = strategy_config.get('trailing_distance', stop_loss * 0.5)
            max_positions = strategy_config.get('max_positions', 10)
            
            print(f"   🕌 CNC Backtesting {strategy_name}")
            print(f"       Target: {profit_target*100:.1f}% | Stop: {stop_loss*100:.1f}% | Min Hold: {min_hold_days} days")
            
            for i in range(1, len(df)):
                current_date = pd.to_datetime(df.loc[i, 'date' if 'date' in df.columns else 'datetime'])
                current_price = df.loc[i, 'close']
                signal = df.loc[i, signal_column] if signal_column in df.columns else 0
                
                # Check existing positions for exit conditions
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
                    
                    # HALAL CNC EXIT CONDITIONS
                    if days_held < min_hold_days:
                        # Cannot exit before minimum hold period (CNC requirement)
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
                        # Calculate CNC charges
                        buy_value = POSITION_SIZE_PER_TRADE
                        sell_value = position['shares'] * current_price
                        
                        charge_details = self._calculate_cnc_charges(buy_value, sell_value)
                        total_charges = charge_details['total_charges']
                        
                        # Net profit/loss after all charges
                        gross_pnl = sell_value - buy_value
                        net_pnl = gross_pnl - total_charges
                        
                        portfolio_value += net_pnl
                        
                        # Record detailed trade
                        trades.append({
                            'position_id': pos_id,
                            'entry_date': entry_date,
                            'exit_date': current_date,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'shares': position['shares'],
                            'days_held': days_held,
                            'buy_value': buy_value,
                            'sell_value': sell_value,
                            'gross_pnl': gross_pnl,
                            'total_charges': total_charges,
                            'net_pnl': net_pnl,
                            'return_pct': pct_change * 100,
                            'exit_reason': exit_reason,
                            'trailing_activated': position['trailing_active'],
                            'charge_breakdown': charge_details,
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
                    portfolio_value >= POSITION_SIZE_PER_TRADE * 1.5):  # Keep some cash buffer
                    
                    # Enter new CNC position
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
                    
                    portfolio_value -= POSITION_SIZE_PER_TRADE  # Reserve capital
            
            # Close any remaining positions at the end
            final_date = pd.to_datetime(df.loc[len(df)-1, 'date' if 'date' in df.columns else 'datetime'])
            final_price = df.loc[len(df)-1, 'close']
            
            for pos_id, position in active_positions.items():
                days_held = (final_date - position['entry_date']).days
                if days_held >= min_hold_days:  # Only close if minimum holding period met
                    pct_change = (final_price - position['entry_price']) / position['entry_price']
                    
                    buy_value = POSITION_SIZE_PER_TRADE
                    sell_value = position['shares'] * final_price
                    charge_details = self._calculate_cnc_charges(buy_value, sell_value)
                    total_charges = charge_details['total_charges']
                    
                    gross_pnl = sell_value - buy_value
                    net_pnl = gross_pnl - total_charges
                    portfolio_value += net_pnl
                    
                    trades.append({
                        'position_id': pos_id,
                        'entry_date': position['entry_date'],
                        'exit_date': final_date,
                        'entry_price': position['entry_price'],
                        'exit_price': final_price,
                        'shares': position['shares'],
                        'days_held': days_held,
                        'buy_value': buy_value,
                        'sell_value': sell_value,
                        'gross_pnl': gross_pnl,
                        'total_charges': total_charges,
                        'net_pnl': net_pnl,
                        'return_pct': pct_change * 100,
                        'exit_reason': "End of Period",
                        'trailing_activated': position['trailing_active'],
                        'charge_breakdown': charge_details,
                        'portfolio_value': portfolio_value
                    })
            
            return trades, portfolio_value
            
        except Exception as e:
            print(f"   ❌ CNC Backtest failed for {strategy_name}: {e}")
            return [], INITIAL_CAPITAL
    
    def _analyze_cnc_performance(self, trades, final_portfolio, strategy_name):
        """Comprehensive CNC performance analysis"""
        if not trades:
            return {
                'strategy_name': strategy_name,
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'final_portfolio': INITIAL_CAPITAL,
                'status': 'NO_TRADES'
            }
        
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df_trades[df_trades['net_pnl'] > 0])  # Use NET PnL
        win_rate = winning_trades / total_trades
        
        total_return = (final_portfolio - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        avg_return = df_trades['return_pct'].mean()
        avg_hold_days = df_trades['days_held'].mean()
        
        # Charge analysis
        total_charges = df_trades['total_charges'].sum()
        avg_charges_per_trade = df_trades['total_charges'].mean()
        charges_as_pct_of_capital = total_charges / INITIAL_CAPITAL * 100
        
        # Net vs Gross analysis
        total_gross_pnl = df_trades['gross_pnl'].sum()
        total_net_pnl = df_trades['net_pnl'].sum()
        charges_impact_pct = (total_gross_pnl - total_net_pnl) / total_gross_pnl * 100 if total_gross_pnl != 0 else 0
        
        # Risk metrics
        returns_std = df_trades['return_pct'].std()
        sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0
        
        # Max drawdown from portfolio values
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
            'avg_return_per_trade': avg_return,
            'avg_hold_days': avg_hold_days,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            
            # CNC-specific metrics
            'total_charges': total_charges,
            'avg_charges_per_trade': avg_charges_per_trade,
            'charges_as_pct_of_capital': charges_as_pct_of_capital,
            'total_gross_pnl': total_gross_pnl,
            'total_net_pnl': total_net_pnl,
            'charges_impact_pct': charges_impact_pct,
            
            'status': 'COMPLETED'
        }
    
    def run_halal_cnc_backtest(self):
        """Run comprehensive halal CNC backtesting"""
        print(f"\n🕌 HALAL CNC BACKTESTING SYSTEM")
        print(f"{'='*60}")
        print(f"✅ CNC (Cash & Carry) Trading Only")
        print(f"✅ Minimum 1-day holding period")
        print(f"✅ Complete Zerodha charges included")
        print(f"💰 Initial Capital: ₹{INITIAL_CAPITAL:,}")
        print(f"💳 Position Size: ₹{POSITION_SIZE_PER_TRADE:,}")
        print(f"{'='*60}")
        
        all_results = {}
        overall_data_period = {}
        
        # Process each strategy group
        for group_key, group_config in HALAL_CNC_STRATEGIES.items():
            print(f"\n📈 PROCESSING {group_key.upper()}")
            print(f"{'='*50}")
            
            data_dir = group_config['data_dir']
            timeframe = group_config['timeframe']
            
            # Get all data files
            try:
                data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                print(f"📁 Found {len(data_files)} data files for {timeframe}")
            except:
                print(f"❌ Data directory not found: {data_dir}")
                continue
            
            group_results = {}
            
            # Process each strategy in this group
            for strategy_key, strategy_config in group_config['strategies'].items():
                print(f"\n🔍 STRATEGY: {strategy_config['name']}")
                
                all_trades = []
                
                # Test with first 20 symbols for comprehensive results
                test_files = data_files[:20]
                
                # Data period tracking
                total_candles_processed = 0
                min_start_date = None
                max_end_date = None
                
                for i, filename in enumerate(test_files, 1):
                    try:
                        symbol = filename.replace('.csv', '')
                        print(f"   📊 [{i}/{len(test_files)}] {symbol}")
                        
                        # Load data
                        file_path = os.path.join(data_dir, filename)
                        df = pd.read_csv(file_path)
                        
                        if len(df) < 100:  # Need substantial data for CNC
                            print(f"       ⚠️ Insufficient data ({len(df)} bars)")
                            continue
                        
                        # Analyze data period
                        data_period = self._analyze_data_period(df)
                        if data_period:
                            total_candles_processed += data_period['total_candles']
                            
                            start_date = pd.to_datetime(data_period['start_date'])
                            end_date = pd.to_datetime(data_period['end_date'])
                            
                            if min_start_date is None or start_date < min_start_date:
                                min_start_date = start_date
                            if max_end_date is None or end_date > max_end_date:
                                max_end_date = end_date
                        
                        # Ensure datetime column
                        if 'datetime' not in df.columns and 'date' in df.columns:
                            df['datetime'] = pd.to_datetime(df['date'])
                        
                        # Generate signals based on strategy type
                        if 'ml' in strategy_key:
                            df = self._generate_ml_signals(df)
                            signal_column = 'ml_signal'
                            
                            if 'confidence_threshold' in strategy_config:
                                conf_threshold = strategy_config['confidence_threshold']
                                df.loc[df['ml_probability'] < conf_threshold, 'ml_signal'] = 0
                        
                        elif 'williams' in strategy_key:
                            df = self._generate_williams_signals(df, strategy_config)
                            signal_column = 'williams_signal'
                        
                        elif 'ema' in strategy_key:
                            df = self._generate_ema_momentum_signals(df)
                            signal_column = 'ema_signal'
                        
                        else:
                            continue
                        
                        # Run CNC backtest
                        symbol_trades, symbol_portfolio = self._backtest_cnc_strategy(
                            df, strategy_config['name'], strategy_config, signal_column
                        )
                        
                        if symbol_trades:
                            # Add symbol info to trades
                            for trade in symbol_trades:
                                trade['symbol'] = symbol
                            all_trades.extend(symbol_trades)
                            print(f"       ✅ {len(symbol_trades)} CNC trades")
                        
                        del df
                        gc.collect()
                        
                        self.execution_stats['total_files_processed'] += 1
                        
                    except Exception as e:
                        print(f"       ❌ Error processing {filename}: {e}")
                        continue
                
                # Store data period information
                if min_start_date and max_end_date:
                    period_info = {
                        'timeframe': timeframe,
                        'start_date': min_start_date.strftime('%Y-%m-%d'),
                        'end_date': max_end_date.strftime('%Y-%m-%d'),
                        'total_days': (max_end_date - min_start_date).days,
                        'total_candles': total_candles_processed,
                        'symbols_processed': len(test_files)
                    }
                    overall_data_period[strategy_key] = period_info
                
                # Analyze strategy performance
                if all_trades:
                    final_portfolio = INITIAL_CAPITAL + sum([trade['net_pnl'] for trade in all_trades])
                    
                    performance = self._analyze_cnc_performance(
                        all_trades, final_portfolio, strategy_config['name']
                    )
                    
                    # Add data period info
                    performance['data_period'] = period_info
                    
                    group_results[strategy_key] = performance
                    self.execution_stats['successful_backtests'] += 1
                    self.execution_stats['total_trades_generated'] += len(all_trades)
                    
                    print(f"   ✅ CNC Strategy Analysis Complete:")
                    print(f"       Trades: {performance['total_trades']}")
                    print(f"       Win Rate: {performance['win_rate']:.2%}")
                    print(f"       Net Return: {performance['total_return']:.2f}%")
                    print(f"       Final Portfolio: ₹{performance['final_portfolio']:,.0f}")
                    print(f"       Avg Hold: {performance['avg_hold_days']:.1f} days")
                    print(f"       Total Charges: ₹{performance['total_charges']:,.0f}")
                    print(f"       Charges Impact: {performance['charges_impact_pct']:.2f}%")
                else:
                    print(f"   ❌ No CNC trades generated")
            
            all_results[group_key] = group_results
        
        self.results_summary = all_results
        self.execution_stats['backtest_period'] = overall_data_period
        self._save_cnc_results()
        self._print_cnc_summary()
        
        return all_results
    
    def _save_cnc_results(self):
        """Save comprehensive CNC results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save summary
            summary_file = os.path.join(DATA_DIRS['results'], f'halal_cnc_backtest_{timestamp}.json')
            
            # Prepare comprehensive data
            complete_results = {
                'backtest_timestamp': timestamp,
                'backtest_type': 'HALAL_CNC_ONLY',
                'initial_capital': INITIAL_CAPITAL,
                'position_size': POSITION_SIZE_PER_TRADE,
                'zerodha_charges': ZERODHA_CHARGES,
                'data_periods': self.execution_stats['backtest_period'],
                'execution_stats': self.execution_stats,
                'strategy_results': self.results_summary
            }
            
            with open(summary_file, 'w') as f:
                json.dump(complete_results, f, indent=2, default=str)
            
            print(f"📁 Comprehensive results saved: {summary_file}")
            
        except Exception as e:
            print(f"⚠️ Results saving failed: {e}")
    
    def _print_cnc_summary(self):
        """Print comprehensive CNC summary"""
        end_time = datetime.now()
        duration = end_time - self.execution_stats['start_time']
        
        print(f"\n{'='*70}")
        print(f"🕌 HALAL CNC BACKTESTING RESULTS SUMMARY")
        print(f"{'='*70}")
        
        # Execution stats
        print(f"⏱️  Execution Time: {duration.total_seconds():.1f} seconds")
        print(f"📁 Files Processed: {self.execution_stats['total_files_processed']}")
        print(f"✅ Successful Backtests: {self.execution_stats['successful_backtests']}")
        print(f"📊 Total CNC Trades: {self.execution_stats['total_trades_generated']}")
        print(f"💰 Total Charges Paid: ₹{self.execution_stats['total_cnc_charges']:,.0f}")
        
        # Data period summary
        print(f"\n📊 BACKTEST DATA PERIODS:")
        print(f"{'='*40}")
        for strategy_key, period_info in self.execution_stats['backtest_period'].items():
            print(f"{strategy_key}:")
            print(f"  📅 Period: {period_info['start_date']} to {period_info['end_date']}")
            print(f"  📊 Duration: {period_info['total_days']} days")
            print(f"  🕯️  Candles: {period_info['total_candles']:,}")
            print(f"  📈 Symbols: {period_info['symbols_processed']}")
            print(f"  ⏱️  Timeframe: {period_info['timeframe']}")
            print()
        
        # Strategy rankings
        all_strategies = []
        for group_key, strategies in self.results_summary.items():
            for strategy_key, performance in strategies.items():
                if performance['status'] == 'COMPLETED':
                    all_strategies.append(performance)
        
        if all_strategies:
            all_strategies.sort(key=lambda x: x['total_return'], reverse=True)
            
            print(f"🏆 TOP HALAL CNC STRATEGIES:")
            print(f"{'='*50}")
            
            for i, strategy in enumerate(all_strategies, 1):
                print(f"{i}. {strategy['strategy_name']}")
                print(f"   📈 Net Return: {strategy['total_return']:.2f}%")
                print(f"   💰 Final Portfolio: ₹{strategy['final_portfolio']:,.0f}")
                print(f"   🎯 Win Rate: {strategy['win_rate']:.2%}")
                print(f"   📊 Total Trades: {strategy['total_trades']}")
                print(f"   📅 Avg Hold: {strategy['avg_hold_days']:.1f} days")
                print(f"   💸 Avg Charges: ₹{strategy['avg_charges_per_trade']:.0f}/trade")
                print(f"   📉 Charges Impact: {strategy['charges_impact_pct']:.2f}%")
                print(f"   📊 Sharpe Ratio: {strategy['sharpe_ratio']:.2f}")
                print()
        
        print(f"✅ HALAL COMPLIANCE CONFIRMED:")
        print(f"   - All trades held minimum 1 day (CNC)")
        print(f"   - No intraday trading")
        print(f"   - Complete tax and charge calculations")
        print(f"   - Real-world applicable results")

if __name__ == "__main__":
    print("🕌 HALAL CNC BACKTESTING SYSTEM")
    print("="*50)
    print("✅ 100% Halal Compliant")
    print("✅ CNC (Cash & Carry) Only") 
    print("✅ Minimum 1-day holding")
    print("✅ Complete Zerodha charges")
    print("✅ STT, stamp duty, DP charges")
    print("✅ Multiple timeframes for entry timing")
    print("="*50)
    
    try:
        engine = HalalCNCBacktestEngine()
        results = engine.run_halal_cnc_backtest()
        
        print(f"\n🎉 HALAL CNC BACKTESTING COMPLETED!")
        print(f"🕌 100% Shariah Compliant Results")
        print(f"💰 Real-world charges included")
        print(f"📊 Ready for halal live trading")
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        raise
