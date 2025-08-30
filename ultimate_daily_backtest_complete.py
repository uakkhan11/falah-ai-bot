# =============================================================================
# ULTIMATE DAILY PROFITABLE TRADING BACKTEST - COMPLETE SYSTEM
# =============================================================================

# PART 1: Multi-timeframe Engine, Trailing System, Signal Generator

import os
import pandas as pd
import numpy as np
import talib
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
from datetime import datetime, timedelta
import random
from scipy import stats
warnings.filterwarnings("ignore")

# =============================================================================
# ULTIMATE CONFIGURATION FOR DAILY PROFITABLE TRADES
# =============================================================================

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data")
}

# Enhanced parameters for daily profitable trading
YEAR_FILTER = 2025
INITIAL_CAPITAL = 100000
MIN_TRADE_SIZE = 2000      # Minimum trade size
MAX_POSITIONS = 2          # Conservative for daily trades
MAX_DAILY_LOSS = 0.02      # 2% max daily loss
TARGET_DAILY_TRADES = 3    # Target 3 trades per day

# Multi-timeframe thresholds
WEEKLY_ADX_MIN = 20        # Weekly trend strength
DAILY_ADX_MIN = 25         # Daily trend strength  
HOURLY_ADX_MIN = 20        # Hourly momentum
M15_ADX_MIN = 15          # 15min entry timing

# Advanced trailing stop parameters
ATR_TRAILING_MULT = 2.5    # ATR multiplier for trailing
BREAKEVEN_RATIO = 1.5      # Move to breakeven after 1.5R profit
PARTIAL_TP_RATIO = 2.0     # First partial TP at 2R
FINAL_TP_RATIO = 4.0       # Final TP at 4R
MAX_HOLD_HOURS = 8         # Max hold time (8 hours for intraday)

# Core features (optimized for daily trading)
CORE_FEATURES = ["multi_tf_adx", "atr_volatility", "volume_surge", "momentum_strength"]

# ML parameters (more conservative for daily trading)
ML_PROBA_THRESHOLD = 0.75  # Higher threshold for daily trades
BASE_RISK = 0.01          # 1% base risk per trade

# =============================================================================
# ULTIMATE MULTI-TIMEFRAME ANALYSIS ENGINE
# =============================================================================

class MultiTimeframeEngine:
    """Advanced multi-timeframe analysis engine"""

    def __init__(self):
        self.timeframes = {
            'weekly': 'W',
            'daily': 'D', 
            'hourly': 'H',
            'm15': '15T'
        }

    def calculate_multi_tf_indicators(self, daily_df, hourly_df, m15_df):
        """Calculate indicators across all timeframes"""
        try:
            # Weekly data (from daily)
            weekly_data = self._resample_to_weekly(daily_df)
            weekly_signals = self._calculate_weekly_signals(weekly_data)

            # Daily signals
            daily_signals = self._calculate_daily_signals(daily_df)

            # Hourly signals 
            hourly_signals = self._calculate_hourly_signals(hourly_df)

            # Combine all timeframe signals to 15-minute data
            enhanced_m15 = self._merge_timeframe_signals(
                m15_df, weekly_signals, daily_signals, hourly_signals
            )

            return enhanced_m15

        except Exception as e:
            print(f"Error in multi-timeframe calculation: {e}")
            return m15_df

    def _resample_to_weekly(self, daily_df):
        """Resample daily data to weekly"""
        try:
            df = daily_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            weekly = df.resample('W-MON').agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            return weekly.reset_index()

        except Exception as e:
            print(f"Error in weekly resampling: {e}")
            return daily_df

    def _calculate_weekly_signals(self, weekly_df):
        """Calculate weekly trend signals"""
        signals = {}

        try:
            # Weekly ADX for major trend
            adx_indicator = ta.trend.ADXIndicator(weekly_df['high'], weekly_df['low'], weekly_df['close'], window=14)
            weekly_adx = adx_indicator.adx()

            # Weekly EMA for trend direction
            weekly_ema50 = ta.trend.ema_indicator(weekly_df['close'], window=50)
            weekly_ema200 = ta.trend.ema_indicator(weekly_df['close'], window=200)

            # Weekly trend strength score
            trend_strength = np.where(
                (weekly_df['close'] > weekly_ema50) & (weekly_ema50 > weekly_ema200) & (weekly_adx > WEEKLY_ADX_MIN),
                2,  # Strong uptrend
                np.where(
                    (weekly_df['close'] > weekly_ema50) & (weekly_adx > WEEKLY_ADX_MIN/2),
                    1,  # Moderate uptrend
                    np.where(
                        (weekly_df['close'] < weekly_ema50) & (weekly_ema50 < weekly_ema200) & (weekly_adx > WEEKLY_ADX_MIN),
                        -2, # Strong downtrend
                        np.where(
                            (weekly_df['close'] < weekly_ema50) & (weekly_adx > WEEKLY_ADX_MIN/2),
                            -1, # Moderate downtrend
                            0   # Sideways
                        )
                    )
                )
            )

            signals['weekly_trend_strength'] = trend_strength
            signals['weekly_adx'] = weekly_adx
            signals['dates'] = weekly_df['date']

            return signals

        except Exception as e:
            print(f"Error in weekly signals: {e}")
            return {'weekly_trend_strength': [0], 'weekly_adx': [20], 'dates': []}

    def _calculate_daily_signals(self, daily_df):
        """Calculate daily momentum signals"""
        signals = {}

        try:
            # Daily ADX
            adx_indicator = ta.trend.ADXIndicator(daily_df['high'], daily_df['low'], daily_df['close'], window=14)
            daily_adx = adx_indicator.adx()

            # Daily momentum indicators
            rsi = ta.momentum.rsi(daily_df['close'], window=14)
            macd_line, macd_signal, macd_hist = talib.MACD(daily_df['close'].values)

            # Volume analysis
            daily_df['volume_sma'] = daily_df['volume'].rolling(20).mean()
            volume_ratio = daily_df['volume'] / daily_df['volume_sma']

            # Daily momentum score
            momentum_score = np.where(
                (daily_adx > DAILY_ADX_MIN) & (rsi > 50) & (rsi < 80) & (macd_hist > 0) & (volume_ratio > 1.2),
                2,  # Strong momentum
                np.where(
                    (daily_adx > DAILY_ADX_MIN/2) & (rsi > 45) & (rsi < 85) & (volume_ratio > 1.0),
                    1,  # Moderate momentum
                    np.where(
                        (daily_adx > DAILY_ADX_MIN) & (rsi < 50) & (rsi > 20) & (macd_hist < 0) & (volume_ratio > 1.2),
                        -2, # Strong bearish momentum
                        np.where(
                            (daily_adx > DAILY_ADX_MIN/2) & (rsi < 55) & (rsi > 15) & (volume_ratio > 1.0),
                            -1, # Moderate bearish momentum
                            0   # Neutral
                        )
                    )
                )
            )

            signals['daily_momentum_score'] = momentum_score
            signals['daily_adx'] = daily_adx
            signals['daily_volume_ratio'] = volume_ratio
            signals['dates'] = daily_df['date']

            return signals

        except Exception as e:
            print(f"Error in daily signals: {e}")
            return {'daily_momentum_score': [0], 'daily_adx': [20], 'daily_volume_ratio': [1.0], 'dates': []}

    def _calculate_hourly_signals(self, hourly_df):
        """Calculate hourly confirmation signals"""
        signals = {}

        try:
            # Hourly ADX
            adx_indicator = ta.trend.ADXIndicator(hourly_df['high'], hourly_df['low'], hourly_df['close'], window=14)
            hourly_adx = adx_indicator.adx()

            # Bollinger Bands for volatility
            bb_indicator = ta.volatility.BollingerBands(hourly_df['close'], window=20)
            bb_upper = bb_indicator.bollinger_hband()
            bb_lower = bb_indicator.bollinger_lband()
            bb_width = (bb_upper - bb_lower) / hourly_df['close']

            # Hourly confirmation score
            confirmation_score = np.where(
                (hourly_adx > HOURLY_ADX_MIN) & (bb_width > bb_width.rolling(20).mean()),
                1,  # Strong confirmation
                np.where(
                    (hourly_adx > HOURLY_ADX_MIN/2),
                    0.5,  # Moderate confirmation
                    0     # Weak confirmation
                )
            )

            signals['hourly_confirmation'] = confirmation_score
            signals['hourly_adx'] = hourly_adx
            signals['hourly_volatility'] = bb_width
            signals['dates'] = hourly_df['date']

            return signals

        except Exception as e:
            print(f"Error in hourly signals: {e}")
            return {'hourly_confirmation': [0], 'hourly_adx': [20], 'hourly_volatility': [0.02], 'dates': []}

    def _merge_timeframe_signals(self, m15_df, weekly_signals, daily_signals, hourly_signals):
        """Merge all timeframe signals to 15-minute data"""
        try:
            m15_df = m15_df.copy()
            m15_df['date'] = pd.to_datetime(m15_df['date'])

            # Forward fill weekly signals
            weekly_df = pd.DataFrame(weekly_signals)
            weekly_df['date'] = pd.to_datetime(weekly_df['date'])
            for col in ['weekly_trend_strength', 'weekly_adx']:
                if col in weekly_df.columns:
                    m15_df = m15_df.merge(weekly_df[['date', col]], on='date', how='left')
                    m15_df[col] = m15_df[col].fillna(method='ffill').fillna(0)

            # Forward fill daily signals  
            daily_df = pd.DataFrame(daily_signals)
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            for col in ['daily_momentum_score', 'daily_adx', 'daily_volume_ratio']:
                if col in daily_df.columns:
                    m15_df = m15_df.merge(daily_df[['date', col]], on='date', how='left')
                    m15_df[col] = m15_df[col].fillna(method='ffill').fillna(0 if 'score' in col or 'adx' in col else 1.0)

            # Forward fill hourly signals
            hourly_df = pd.DataFrame(hourly_signals) 
            hourly_df['date'] = pd.to_datetime(hourly_df['date'])
            for col in ['hourly_confirmation', 'hourly_adx', 'hourly_volatility']:
                if col in hourly_df.columns:
                    m15_df = m15_df.merge(hourly_df[['date', col]], on='date', how='left')
                    m15_df[col] = m15_df[col].fillna(method='ffill').fillna(0 if 'confirmation' in col or 'adx' in col else 0.02)

            return m15_df

        except Exception as e:
            print(f"Error merging timeframe signals: {e}")
            return m15_df

# =============================================================================
# ULTIMATE TRAILING STOP SYSTEM
# =============================================================================

class UltimateTrailingSystem:
    """Advanced trailing stop system with multiple methods"""

    def __init__(self):
        self.trailing_methods = {
            'atr_dynamic': self._atr_dynamic_trail,
            'parabolic_sar': self._parabolic_sar_trail,
            'support_resistance': self._support_resistance_trail,
            'volatility_adaptive': self._volatility_adaptive_trail
        }

    def calculate_trailing_stop(self, position, current_bar, method='atr_dynamic'):
        """Calculate trailing stop using specified method"""
        try:
            if method in self.trailing_methods:
                return self.trailing_methods[method](position, current_bar)
            else:
                return self._atr_dynamic_trail(position, current_bar)
        except Exception as e:
            print(f"Error in trailing stop calculation: {e}")
            return position.get('trail_stop', position['entry_price'] * 0.98)

    def _atr_dynamic_trail(self, position, current_bar):
        """ATR-based dynamic trailing stop"""
        try:
            current_price = current_bar['close']
            atr = current_bar.get('atr', current_price * 0.02)

            # Calculate ATR-based trail distance
            trail_distance = ATR_TRAILING_MULT * atr

            if position['side'] == 'long':
                new_trail_stop = current_price - trail_distance

                # Only move trail stop up, never down
                if 'trail_stop' not in position:
                    position['trail_stop'] = new_trail_stop
                else:
                    position['trail_stop'] = max(position['trail_stop'], new_trail_stop)

            else:  # short position
                new_trail_stop = current_price + trail_distance

                if 'trail_stop' not in position:
                    position['trail_stop'] = new_trail_stop
                else:
                    position['trail_stop'] = min(position['trail_stop'], new_trail_stop)

            return position['trail_stop']

        except Exception as e:
            print(f"Error in ATR trailing: {e}")
            return position.get('trail_stop', current_bar['close'] * 0.98)

    def _parabolic_sar_trail(self, position, current_bar):
        """Parabolic SAR trailing stop"""
        try:
            # Simplified Parabolic SAR calculation
            if 'sar' not in position:
                position['sar'] = position['entry_price']
                position['af'] = 0.02  # Acceleration factor
                position['ep'] = current_bar['high'] if position['side'] == 'long' else current_bar['low']

            af = position['af']
            ep = position['ep']
            sar = position['sar']

            if position['side'] == 'long':
                if current_bar['high'] > ep:
                    ep = current_bar['high']
                    af = min(af + 0.02, 0.20)

                new_sar = sar + af * (ep - sar)

                # SAR cannot be above the low of current or previous bar
                new_sar = min(new_sar, current_bar['low'])

            else:  # short
                if current_bar['low'] < ep:
                    ep = current_bar['low']
                    af = min(af + 0.02, 0.20)

                new_sar = sar + af * (ep - sar)
                new_sar = max(new_sar, current_bar['high'])

            position['sar'] = new_sar
            position['af'] = af
            position['ep'] = ep
            position['trail_stop'] = new_sar

            return new_sar

        except Exception as e:
            print(f"Error in Parabolic SAR trailing: {e}")
            return position.get('trail_stop', current_bar['close'] * 0.98)

    def _support_resistance_trail(self, position, current_bar):
        """Support/Resistance based trailing stop"""
        try:
            # This would need historical data to find support/resistance levels
            # Simplified version using recent highs/lows
            lookback = getattr(current_bar, 'lookback_data', [])

            if len(lookback) < 10:
                return self._atr_dynamic_trail(position, current_bar)

            if position['side'] == 'long':
                # Find recent support level
                recent_lows = [bar['low'] for bar in lookback[-20:]]
                support_level = np.percentile(recent_lows, 25)  # 25th percentile as support

                new_trail_stop = max(support_level, position.get('trail_stop', support_level))

            else:  # short
                recent_highs = [bar['high'] for bar in lookback[-20:]]
                resistance_level = np.percentile(recent_highs, 75)  # 75th percentile as resistance

                new_trail_stop = min(resistance_level, position.get('trail_stop', resistance_level))

            position['trail_stop'] = new_trail_stop
            return new_trail_stop

        except Exception as e:
            print(f"Error in S/R trailing: {e}")
            return self._atr_dynamic_trail(position, current_bar)

    def _volatility_adaptive_trail(self, position, current_bar):
        """Volatility-adaptive trailing stop"""
        try:
            current_price = current_bar['close']

            # Get recent volatility
            volatility = current_bar.get('hourly_volatility', 0.02)

            # Adjust trail distance based on volatility
            base_trail = 0.015  # 1.5% base
            vol_adjustment = volatility * 2  # Scale volatility

            trail_distance = base_trail + vol_adjustment
            trail_distance = min(trail_distance, 0.05)  # Cap at 5%

            if position['side'] == 'long':
                new_trail_stop = current_price * (1 - trail_distance)
                position['trail_stop'] = max(position.get('trail_stop', new_trail_stop), new_trail_stop)
            else:
                new_trail_stop = current_price * (1 + trail_distance) 
                position['trail_stop'] = min(position.get('trail_stop', new_trail_stop), new_trail_stop)

            return position['trail_stop']

        except Exception as e:
            print(f"Error in volatility adaptive trailing: {e}")
            return self._atr_dynamic_trail(position, current_bar)

# =============================================================================
# ULTIMATE SIGNAL GENERATION SYSTEM
# =============================================================================

class UltimateSignalGenerator:
    """Advanced signal generation for daily profitable trades"""

    def __init__(self, mt_engine, trail_system):
        self.mt_engine = mt_engine
        self.trail_system = trail_system

    def generate_daily_signals(self, df):
        """Generate multiple signals per day for consistent trading"""

        signals = []

        try:
            # Strategy 1: Multi-timeframe breakout
            breakout_signals = self._generate_breakout_signals(df)
            signals.extend(breakout_signals)

            # Strategy 2: Mean reversion in trends
            reversion_signals = self._generate_reversion_signals(df)
            signals.extend(reversion_signals)

            # Strategy 3: Momentum continuation
            momentum_signals = self._generate_momentum_signals(df)
            signals.extend(momentum_signals)

            # Strategy 4: Volume surge trades
            volume_signals = self._generate_volume_signals(df)
            signals.extend(volume_signals)

            # Combine and prioritize signals
            df = self._combine_signals(df, signals)

            return df

        except Exception as e:
            print(f"Error in signal generation: {e}")
            return df

    def _generate_breakout_signals(self, df):
        """Multi-timeframe breakout signals"""
        signals = []

        try:
            # Donchian breakout with multi-timeframe confirmation
            df['donchian_high_20'] = df['high'].rolling(20).max()
            df['donchian_low_20'] = df['low'].rolling(20).min()

            for i in range(50, len(df)):
                current_bar = df.iloc[i]

                # Breakout condition
                breakout_long = (
                    current_bar['close'] > current_bar['donchian_high_20'] and
                    current_bar.get('weekly_trend_strength', 0) >= 1 and
                    current_bar.get('daily_momentum_score', 0) >= 1 and
                    current_bar.get('hourly_confirmation', 0) >= 0.5 and
                    current_bar.get('daily_volume_ratio', 1.0) > 1.5
                )

                breakout_short = (
                    current_bar['close'] < current_bar['donchian_low_20'] and
                    current_bar.get('weekly_trend_strength', 0) <= -1 and
                    current_bar.get('daily_momentum_score', 0) <= -1 and
                    current_bar.get('hourly_confirmation', 0) >= 0.5 and
                    current_bar.get('daily_volume_ratio', 1.0) > 1.5
                )

                if breakout_long:
                    signals.append({
                        'index': i,
                        'type': 'breakout_long',
                        'strength': 3,
                        'entry_price': current_bar['close'],
                        'stop_loss': current_bar['donchian_low_20'],
                        'take_profit': current_bar['close'] + 2 * (current_bar['close'] - current_bar['donchian_low_20'])
                    })

                elif breakout_short:
                    signals.append({
                        'index': i,
                        'type': 'breakout_short', 
                        'strength': 3,
                        'entry_price': current_bar['close'],
                        'stop_loss': current_bar['donchian_high_20'],
                        'take_profit': current_bar['close'] - 2 * (current_bar['donchian_high_20'] - current_bar['close'])
                    })

        except Exception as e:
            print(f"Error in breakout signals: {e}")

        return signals

    def _generate_reversion_signals(self, df):
        """Mean reversion signals in trending markets"""
        signals = []

        try:
            # RSI with Bollinger Bands for reversion in trends
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            df['bb_middle'] = bb_indicator.bollinger_mavg()

            for i in range(50, len(df)):
                current_bar = df.iloc[i]
                prev_bar = df.iloc[i-1]

                # Mean reversion long (buy dip in uptrend)
                reversion_long = (
                    current_bar.get('weekly_trend_strength', 0) >= 1 and  # Weekly uptrend
                    current_bar['rsi'] < 35 and prev_bar['rsi'] >= 35 and  # RSI oversold reversal
                    current_bar['close'] < current_bar['bb_lower'] and  # Below BB lower
                    current_bar.get('daily_momentum_score', 0) >= 0  # Daily not bearish
                )

                # Mean reversion short (sell bounce in downtrend)
                reversion_short = (
                    current_bar.get('weekly_trend_strength', 0) <= -1 and  # Weekly downtrend
                    current_bar['rsi'] > 65 and prev_bar['rsi'] <= 65 and  # RSI overbought reversal
                    current_bar['close'] > current_bar['bb_upper'] and  # Above BB upper
                    current_bar.get('daily_momentum_score', 0) <= 0  # Daily not bullish
                )

                if reversion_long:
                    signals.append({
                        'index': i,
                        'type': 'reversion_long',
                        'strength': 2,
                        'entry_price': current_bar['close'],
                        'stop_loss': current_bar['close'] * 0.985,  # Tight stop
                        'take_profit': current_bar['bb_middle']
                    })

                elif reversion_short:
                    signals.append({
                        'index': i,
                        'type': 'reversion_short',
                        'strength': 2,
                        'entry_price': current_bar['close'],
                        'stop_loss': current_bar['close'] * 1.015,  # Tight stop
                        'take_profit': current_bar['bb_middle']
                    })

        except Exception as e:
            print(f"Error in reversion signals: {e}")

        return signals

    def _generate_momentum_signals(self, df):
        """Momentum continuation signals"""
        signals = []

        try:
            # MACD and ADX for momentum
            macd_line, macd_signal, macd_hist = talib.MACD(df['close'].values)
            df['macd_hist'] = macd_hist

            # ADX
            adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            df['adx'] = adx_indicator.adx()

            for i in range(50, len(df)):
                current_bar = df.iloc[i]

                # Momentum long
                momentum_long = (
                    current_bar.get('daily_momentum_score', 0) >= 2 and  # Strong daily momentum
                    current_bar.get('macd_hist', 0) > 0 and  # MACD histogram positive
                    current_bar.get('adx', 0) > M15_ADX_MIN and  # Trending market
                    current_bar.get('hourly_confirmation', 0) >= 0.5
                )

                # Momentum short
                momentum_short = (
                    current_bar.get('daily_momentum_score', 0) <= -2 and  # Strong bearish momentum
                    current_bar.get('macd_hist', 0) < 0 and  # MACD histogram negative
                    current_bar.get('adx', 0) > M15_ADX_MIN and  # Trending market
                    current_bar.get('hourly_confirmation', 0) >= 0.5
                )

                if momentum_long:
                    atr = current_bar.get('atr', current_bar['close'] * 0.02)
                    signals.append({
                        'index': i,
                        'type': 'momentum_long',
                        'strength': 2,
                        'entry_price': current_bar['close'],
                        'stop_loss': current_bar['close'] - 2 * atr,
                        'take_profit': current_bar['close'] + 3 * atr
                    })

                elif momentum_short:
                    atr = current_bar.get('atr', current_bar['close'] * 0.02)
                    signals.append({
                        'index': i,
                        'type': 'momentum_short',
                        'strength': 2,
                        'entry_price': current_bar['close'],
                        'stop_loss': current_bar['close'] + 2 * atr,
                        'take_profit': current_bar['close'] - 3 * atr
                    })

        except Exception as e:
            print(f"Error in momentum signals: {e}")

        return signals

    def _generate_volume_signals(self, df):
        """Volume surge signals"""
        signals = []

        try:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['price_change'] = df['close'].pct_change()

            for i in range(50, len(df)):
                current_bar = df.iloc[i]

                # Volume surge long
                volume_long = (
                    current_bar.get('volume_ratio', 1.0) > 2.0 and  # 2x average volume
                    current_bar.get('price_change', 0) > 0.005 and  # Price up 0.5%+
                    current_bar.get('weekly_trend_strength', 0) >= 0 and  # Not in downtrend
                    current_bar.get('daily_momentum_score', 0) >= 0
                )

                # Volume surge short
                volume_short = (
                    current_bar.get('volume_ratio', 1.0) > 2.0 and  # 2x average volume
                    current_bar.get('price_change', 0) < -0.005 and  # Price down 0.5%+
                    current_bar.get('weekly_trend_strength', 0) <= 0 and  # Not in uptrend
                    current_bar.get('daily_momentum_score', 0) <= 0
                )

                if volume_long:
                    signals.append({
                        'index': i,
                        'type': 'volume_long',
                        'strength': 1,
                        'entry_price': current_bar['close'],
                        'stop_loss': current_bar['close'] * 0.99,
                        'take_profit': current_bar['close'] * 1.02
                    })

                elif volume_short:
                    signals.append({
                        'index': i,
                        'type': 'volume_short',
                        'strength': 1,
                        'entry_price': current_bar['close'],
                        'stop_loss': current_bar['close'] * 1.01,
                        'take_profit': current_bar['close'] * 0.98
                    })

        except Exception as e:
            print(f"Error in volume signals: {e}")

        return signals

    def _combine_signals(self, df, signals):
        """Combine and prioritize all signals"""
        try:
            df['entry_signal'] = 0
            df['signal_type'] = ''
            df['signal_strength'] = 0
            df['suggested_sl'] = 0
            df['suggested_tp'] = 0

            # Sort signals by strength (highest first)
            signals.sort(key=lambda x: x['strength'], reverse=True)

            # Apply signals to dataframe
            for signal in signals:
                idx = signal['index']
                if idx < len(df):
                    # Only apply if no stronger signal exists
                    if df.iloc[idx]['entry_signal'] == 0 or df.iloc[idx]['signal_strength'] < signal['strength']:
                        df.iloc[idx, df.columns.get_loc('entry_signal')] = 1
                        df.iloc[idx, df.columns.get_loc('signal_type')] = signal['type']
                        df.iloc[idx, df.columns.get_loc('signal_strength')] = signal['strength']
                        df.iloc[idx, df.columns.get_loc('suggested_sl')] = signal['stop_loss']
                        df.iloc[idx, df.columns.get_loc('suggested_tp')] = signal['take_profit']

            return df

        except Exception as e:
            print(f"Error combining signals: {e}")
            return df

# =============================================================================
# Continue with the rest of the code...
# =============================================================================

# ... (The code continues with the enhanced backtesting engine, ML system, etc.)

print("ULTIMATE BACKTESTING FRAMEWORK - PART 1 COMPLETE")
print("=" * 60)
print("✓ Multi-timeframe analysis engine")
print("✓ Advanced trailing stop system") 
print("✓ Daily signal generation system")
print("✓ Multiple strategy combinations")
print()
print("This is Part 1 of the ultimate framework.")
print("Part 2 will include the enhanced backtesting engine and ML system.")


# PART 2: ML System, Position Manager, Backtesting Engine

# =============================================================================
# ULTIMATE BACKTESTING FRAMEWORK - PART 2
# =============================================================================

# =============================================================================
# ENHANCED ML SYSTEM FOR DAILY TRADES
# =============================================================================

class UltimateMLSystem:
    """Enhanced ML system optimized for daily profitable trades"""

    def __init__(self):
        self.features = CORE_FEATURES
        self.model = None
        self.scaler = None

    def create_ml_features(self, df):
        """Create ML features from multi-timeframe data"""
        try:
            # Multi-timeframe ADX feature
            df['multi_tf_adx'] = (
                df.get('weekly_adx', 20) * 0.4 +
                df.get('daily_adx', 20) * 0.3 + 
                df.get('hourly_adx', 20) * 0.2 +
                df.get('adx', 20) * 0.1
            )

            # ATR volatility feature
            df['atr_volatility'] = df.get('atr', df['close'] * 0.02) / df['close']

            # Volume surge feature
            df['volume_surge'] = (
                df.get('daily_volume_ratio', 1.0) * 0.6 +
                df.get('volume_ratio', 1.0) * 0.4
            )

            # Momentum strength feature
            df['momentum_strength'] = (
                df.get('weekly_trend_strength', 0) * 0.4 +
                df.get('daily_momentum_score', 0) * 0.4 +
                df.get('hourly_confirmation', 0) * 0.2
            )

            return df

        except Exception as e:
            print(f"Error creating ML features: {e}")
            return df

    def create_labels(self, df, lookforward=2):
        """Create labels for daily trading (shorter lookforward)"""
        try:
            # Create multiple target scenarios
            df['future_return_2'] = df['close'].shift(-2) / df['close'] - 1
            df['future_return_4'] = df['close'].shift(-4) / df['close'] - 1

            # Create label based on profitability within next few bars
            df['profitable_trade'] = (
                (df['future_return_2'] > 0.01) |  # 1% profit in 2 bars
                (df['future_return_4'] > 0.015)   # 1.5% profit in 4 bars
            ).astype(int)

            # Remove future data to avoid look-ahead bias
            df = df[:-lookforward].copy()

            return df

        except Exception as e:
            print(f"Error creating labels: {e}")
            return df

    def train_model(self, df):
        """Train enhanced ML model for daily trades"""
        try:
            # Create features and labels
            df = self.create_ml_features(df)
            df = self.create_labels(df)

            # Clean data
            feature_cols = [col for col in self.features if col in df.columns]
            if not feature_cols:
                print("No valid features found")
                return None, {}

            df_clean = df[feature_cols + ['profitable_trade']].dropna()

            if len(df_clean) < 100:
                print(f"Insufficient data: {len(df_clean)} samples")
                return None, {}

            X = df_clean[feature_cols]
            y = df_clean['profitable_trade']

            # Check label distribution
            pos_rate = y.mean()
            if pos_rate < 0.1 or pos_rate > 0.9:
                print(f"Imbalanced labels: {pos_rate:.2%} positive")
                return None, {}

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False, random_state=42
            )

            # Enhanced XGBoost for daily trading
            self.model = XGBClassifier(
                max_depth=4,              # Slightly more complex for daily patterns
                learning_rate=0.08,       # Faster learning for daily signals
                subsample=0.8,            # More data for daily patterns
                colsample_bytree=0.8,     # Use more features
                n_estimators=50,          # Moderate complexity
                reg_alpha=0.5,            # L1 regularization
                reg_lambda=0.5,           # L2 regularization
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            # Train with early stopping
            eval_set = [(X_test, y_test)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False
            )

            # Evaluate model
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)

            metrics = {
                'train_accuracy': accuracy_score(y_train, train_pred),
                'test_accuracy': accuracy_score(y_test, test_pred),
                'train_precision': precision_score(y_train, train_pred, zero_division=0),
                'test_precision': precision_score(y_test, test_pred, zero_division=0),
                'feature_count': len(feature_cols),
                'positive_rate': pos_rate,
                'training_samples': len(X_train)
            }

            print(f"ML Model trained: {metrics['test_accuracy']:.3f} accuracy, {metrics['test_precision']:.3f} precision")

            return self.model, metrics

        except Exception as e:
            print(f"Error training ML model: {e}")
            return None, {}

    def predict(self, df):
        """Make predictions for trading signals"""
        try:
            if self.model is None:
                return np.ones(len(df)) * 0.5  # Neutral prediction

            df = self.create_ml_features(df)
            feature_cols = [col for col in self.features if col in df.columns]

            if not feature_cols:
                return np.ones(len(df)) * 0.5

            X = df[feature_cols].fillna(method='ffill').fillna(0)

            # Get prediction probabilities
            probabilities = self.model.predict_proba(X)[:, 1]

            return probabilities

        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return np.ones(len(df)) * 0.5

# =============================================================================
# ULTIMATE POSITION MANAGEMENT SYSTEM
# =============================================================================

class UltimatePositionManager:
    """Advanced position management for daily profitable trades"""

    def __init__(self, initial_capital, trail_system):
        self.initial_capital = initial_capital
        self.trail_system = trail_system
        self.daily_pnl = 0
        self.daily_trade_count = 0
        self.max_daily_loss_hit = False

    def calculate_position_size(self, signal_strength, ml_probability, available_capital, atr, price):
        """Calculate optimal position size"""
        try:
            # Base risk calculation
            base_risk = BASE_RISK

            # Adjust risk based on signal strength
            strength_multiplier = {
                3: 1.5,   # Strong signals get bigger size
                2: 1.0,   # Normal signals
                1: 0.5    # Weak signals get smaller size
            }.get(signal_strength, 0.5)

            # Adjust risk based on ML probability
            ml_multiplier = min(2.0, max(0.3, (ml_probability - 0.5) * 4))

            # Final risk percentage
            risk_pct = base_risk * strength_multiplier * ml_multiplier
            risk_pct = min(risk_pct, 0.03)  # Cap at 3%

            # Position value based on risk
            risk_amount = available_capital * risk_pct
            stop_distance = 2 * atr  # 2x ATR stop

            # Calculate shares
            shares = risk_amount / stop_distance
            position_value = shares * price

            # Ensure minimum trade size
            if position_value < MIN_TRADE_SIZE:
                shares = MIN_TRADE_SIZE / price
                position_value = MIN_TRADE_SIZE

            return shares, position_value

        except Exception as e:
            print(f"Error calculating position size: {e}")
            return MIN_TRADE_SIZE / price, MIN_TRADE_SIZE

    def should_enter_trade(self, signal_strength, ml_probability, current_time):
        """Determine if we should enter a new trade"""
        try:
            # Check daily limits
            if self.max_daily_loss_hit:
                return False, "Daily loss limit reached"

            if self.daily_trade_count >= TARGET_DAILY_TRADES * 2:  # Max 6 trades per day
                return False, "Daily trade limit reached"

            # Check signal quality
            if signal_strength < 1:
                return False, "Signal too weak"

            if ml_probability < ML_PROBA_THRESHOLD:
                return False, f"ML probability too low: {ml_probability:.3f}"

            # Check market hours (9:30 AM to 3:00 PM IST)
            try:
                if isinstance(current_time, str):
                    current_time = pd.to_datetime(current_time)

                hour = current_time.hour
                if hour < 9 or hour > 15:
                    return False, "Outside market hours"

            except:
                pass  # If time parsing fails, continue

            return True, "Trade approved"

        except Exception as e:
            print(f"Error in trade approval: {e}")
            return False, "Error in trade approval"

    def manage_position(self, position, current_bar, bar_index):
        """Manage existing position with advanced logic"""
        try:
            current_price = current_bar['close']
            entry_price = position['entry_price']
            side = position.get('side', 'long')

            # Calculate current P&L
            if side == 'long':
                unrealized_pnl = (current_price - entry_price) * position['shares']
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                unrealized_pnl = (entry_price - current_price) * position['shares']
                pnl_pct = (entry_price - current_price) / entry_price

            # Update position tracking
            position['unrealized_pnl'] = unrealized_pnl
            position['pnl_pct'] = pnl_pct
            position['bars_held'] = bar_index - position['entry_bar']

            # Breakeven management
            if abs(pnl_pct) >= BREAKEVEN_RATIO * BASE_RISK and not position.get('moved_to_breakeven', False):
                position['initial_stop'] = entry_price  # Move stop to breakeven
                position['moved_to_breakeven'] = True
                print(f"Moved stop to breakeven for position at {pnl_pct:.2%} profit")

            # Partial profit taking
            if not position.get('partial_taken', False) and abs(pnl_pct) >= PARTIAL_TP_RATIO * BASE_RISK:
                position['take_partial'] = True
                position['partial_taken'] = True
                print(f"Taking partial profits at {pnl_pct:.2%}")

            # Update trailing stop
            trail_stop = self.trail_system.calculate_trailing_stop(position, current_bar)

            # Exit conditions
            exit_reason = None
            exit_price = current_price

            # 1. Trailing stop hit
            if side == 'long' and current_price <= trail_stop:
                exit_reason = 'Trailing Stop'
                exit_price = trail_stop
            elif side == 'short' and current_price >= trail_stop:
                exit_reason = 'Trailing Stop' 
                exit_price = trail_stop

            # 2. Initial stop loss (if not moved to breakeven)
            elif not position.get('moved_to_breakeven', False):
                initial_stop = position.get('initial_stop', entry_price * (0.98 if side == 'long' else 1.02))
                if side == 'long' and current_price <= initial_stop:
                    exit_reason = 'Initial Stop Loss'
                    exit_price = initial_stop
                elif side == 'short' and current_price >= initial_stop:
                    exit_reason = 'Initial Stop Loss'
                    exit_price = initial_stop

            # 3. Final take profit
            elif abs(pnl_pct) >= FINAL_TP_RATIO * BASE_RISK:
                exit_reason = 'Final Take Profit'

            # 4. Maximum hold time
            elif position['bars_held'] >= MAX_HOLD_HOURS * 4:  # 4 bars per hour (15min)
                exit_reason = 'Time Exit'

            # 5. End of day exit (3:15 PM IST)
            try:
                if isinstance(current_bar.get('date'), str):
                    current_time = pd.to_datetime(current_bar['date'])
                    if current_time.hour >= 15 and current_time.minute >= 15:
                        exit_reason = 'End of Day'
            except:
                pass

            if exit_reason:
                return self._execute_exit(position, exit_price, exit_reason, current_bar)

            return None  # Keep position open

        except Exception as e:
            print(f"Error managing position: {e}")
            return None

    def _execute_exit(self, position, exit_price, exit_reason, current_bar):
        """Execute position exit"""
        try:
            entry_price = position['entry_price']
            shares = position['shares']
            side = position.get('side', 'long')

            # Calculate slippage
            volatility = current_bar.get('hourly_volatility', 0.02)
            slippage = 0.0005 + volatility * 0.5  # Base + volatility component
            slippage = min(slippage, 0.01)  # Cap at 1%

            if side == 'long':
                final_exit_price = exit_price * (1 - slippage)
            else:
                final_exit_price = exit_price * (1 + slippage)

            # Calculate final P&L
            if side == 'long':
                gross_pnl = (final_exit_price - entry_price) * shares
            else:
                gross_pnl = (entry_price - final_exit_price) * shares

            # Calculate costs
            entry_cost = entry_price * shares * 0.0005  # Entry commission
            exit_cost = final_exit_price * shares * 0.0005  # Exit commission
            slippage_cost = abs(exit_price - final_exit_price) * shares
            total_costs = entry_cost + exit_cost + slippage_cost

            net_pnl = gross_pnl - total_costs

            # Update daily tracking
            self.daily_pnl += net_pnl
            self.daily_trade_count += 1

            # Check daily loss limit
            if self.daily_pnl <= -self.initial_capital * MAX_DAILY_LOSS:
                self.max_daily_loss_hit = True
                print(f"Daily loss limit hit: {self.daily_pnl:.2f}")

            trade_record = {
                'entry_date': position['entry_date'],
                'exit_date': current_bar.get('date', ''),
                'entry_price': entry_price,
                'exit_price': final_exit_price,
                'shares': shares,
                'side': side,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'return_pct': net_pnl / (entry_price * shares),
                'exit_reason': exit_reason,
                'bars_held': position['bars_held'],
                'slippage_cost': slippage_cost,
                'commission_cost': entry_cost + exit_cost,
                'signal_type': position.get('signal_type', ''),
                'signal_strength': position.get('signal_strength', 1)
            }

            return trade_record

        except Exception as e:
            print(f"Error executing exit: {e}")
            return None

    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0
        self.daily_trade_count = 0
        self.max_daily_loss_hit = False

# =============================================================================
# ULTIMATE BACKTESTING ENGINE
# =============================================================================

class UltimateBacktestEngine:
    """Ultimate backtesting engine for daily profitable trades"""

    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.mt_engine = MultiTimeframeEngine()
        self.trail_system = UltimateTrailingSystem()
        self.signal_generator = UltimateSignalGenerator(self.mt_engine, self.trail_system)
        self.ml_system = UltimateMLSystem()
        self.position_manager = UltimatePositionManager(initial_capital, self.trail_system)

    def run_backtest(self, symbol):
        """Run complete backtest for a symbol"""
        try:
            print(f"\n=== BACKTESTING {symbol} ===")

            # Load data
            daily_df, hourly_df, m15_df = self._load_data(symbol)
            if m15_df is None or len(m15_df) < 1000:
                print(f"Insufficient data for {symbol}")
                return []

            # Prepare multi-timeframe data
            print("Preparing multi-timeframe analysis...")
            m15_df = self.mt_engine.calculate_multi_tf_indicators(daily_df, hourly_df, m15_df)

            # Generate trading signals
            print("Generating trading signals...")
            m15_df = self.signal_generator.generate_daily_signals(m15_df)

            # Train ML model on first 70% of data
            split_point = int(len(m15_df) * 0.7)
            train_data = m15_df.iloc[:split_point].copy()

            print("Training ML model...")
            ml_model, ml_metrics = self.ml_system.train_model(train_data)

            if ml_model is None:
                print(f"Failed to train ML model for {symbol}")
                return []

            # Run walk-forward backtest on remaining 30%
            test_data = m15_df.iloc[split_point:].copy()
            print(f"Running backtest on {len(test_data)} bars...")

            trades = self._execute_backtest(test_data, symbol)

            if trades:
                print(f"Completed: {len(trades)} trades executed")
                return trades
            else:
                print("No trades executed")
                return []

        except Exception as e:
            print(f"Error backtesting {symbol}: {e}")
            return []

    def _load_data(self, symbol):
        """Load and prepare data"""
        try:
            daily = pd.read_csv(os.path.join(DATA_PATHS['daily'], f"{symbol}.csv"))
            hourly = pd.read_csv(os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv"))
            m15 = pd.read_csv(os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"))

            # Filter for 2025
            for df in [daily, hourly, m15]:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'].dt.year == YEAR_FILTER].reset_index(drop=True)

            # Add basic indicators
            for df in [daily, hourly, m15]:
                df = self._add_basic_indicators(df)

            return daily, hourly, m15

        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return None, None, None

    def _add_basic_indicators(self, df):
        """Add basic indicators to dataframe"""
        try:
            if len(df) < 50:
                return df

            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

            # Volume ratio
            df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            # ADX
            try:
                adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
                df['adx'] = adx_indicator.adx()
            except:
                df['adx'] = 20

            # Fill NaN values
            df['atr'] = df['atr'].fillna(df['close'] * 0.02)
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
            df['adx'] = df['adx'].fillna(20)

            return df

        except Exception as e:
            print(f"Error adding indicators: {e}")
            return df

    def _execute_backtest(self, df, symbol):
        """Execute the actual backtesting"""
        try:
            cash = self.initial_capital
            positions = {}
            trades = []
            daily_reset_date = None

            # Get ML predictions
            ml_predictions = self.ml_system.predict(df)

            for i in range(100, len(df)):  # Start after sufficient lookback
                current_bar = df.iloc[i]
                current_date = current_bar['date']
                current_price = current_bar['close']

                # Reset daily stats if new day
                if daily_reset_date != current_date.date():
                    self.position_manager.reset_daily_stats()
                    daily_reset_date = current_date.date()

                # Manage existing positions
                positions_to_close = []
                for pos_id, position in positions.items():
                    exit_trade = self.position_manager.manage_position(position, current_bar, i)
                    if exit_trade:
                        exit_trade['symbol'] = symbol
                        trades.append(exit_trade)
                        positions_to_close.append(pos_id)

                        # Update cash
                        cash += exit_trade['net_pnl'] + (position['shares'] * position['entry_price'])

                # Remove closed positions
                for pos_id in positions_to_close:
                    del positions[pos_id]

                # Check for new entry signals
                if (current_bar.get('entry_signal', 0) == 1 and 
                    len(positions) < MAX_POSITIONS and 
                    cash > MIN_TRADE_SIZE):

                    signal_strength = current_bar.get('signal_strength', 1)
                    ml_prob = ml_predictions[i] if i < len(ml_predictions) else 0.5

                    # Check if we should enter trade
                    should_enter, reason = self.position_manager.should_enter_trade(
                        signal_strength, ml_prob, current_date
                    )

                    if should_enter:
                        # Calculate position size
                        atr = current_bar.get('atr', current_price * 0.02)
                        shares, position_value = self.position_manager.calculate_position_size(
                            signal_strength, ml_prob, cash, atr, current_price
                        )

                        if cash >= position_value:
                            # Create new position
                            position = {
                                'entry_date': current_date,
                                'entry_price': current_price,
                                'shares': shares,
                                'side': 'long',  # Simplified to long only for now
                                'entry_bar': i,
                                'signal_type': current_bar.get('signal_type', ''),
                                'signal_strength': signal_strength,
                                'ml_probability': ml_prob,
                                'initial_stop': current_price * 0.98  # 2% initial stop
                            }

                            positions[len(positions)] = position
                            cash -= position_value

                            print(f"Entered {position['signal_type']} at {current_price:.2f}, ML prob: {ml_prob:.3f}")

                # Stop if we hit max daily loss
                if self.position_manager.max_daily_loss_hit:
                    break

            return trades

        except Exception as e:
            print(f"Error executing backtest: {e}")
            return []

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_ultimate_daily_backtest():
    """Main function to run the ultimate daily profitable trading backtest"""

    print("🚀 ULTIMATE DAILY PROFITABLE TRADING BACKTEST")
    print("=" * 65)
    print(f"Target: {TARGET_DAILY_TRADES} profitable trades per day")
    print(f"Max daily loss: {MAX_DAILY_LOSS:.1%}")
    print(f"Max positions: {MAX_POSITIONS}")
    print(f"ML threshold: {ML_PROBA_THRESHOLD}")
    print()

    # Get symbols
    try:
        symbols = [f.replace('.csv', '') for f in os.listdir(DATA_PATHS['daily']) 
                  if f.endswith('.csv')][:15]  # Limit to 15 symbols
        print(f"Found {len(symbols)} symbols to test")
    except:
        print("Error loading symbols")
        return

    # Initialize backtesting engine
    engine = UltimateBacktestEngine()

    all_results = []
    all_trades = []
    successful_symbols = 0

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

        try:
            trades = engine.run_backtest(symbol)

            if trades and len(trades) > 5:  # Need minimum trades for analysis
                # Calculate statistics
                trades_df = pd.DataFrame(trades)

                stats = {
                    'symbol': symbol,
                    'total_trades': len(trades),
                    'winning_trades': len(trades_df[trades_df['net_pnl'] > 0]),
                    'win_rate': len(trades_df[trades_df['net_pnl'] > 0]) / len(trades) * 100,
                    'total_pnl': trades_df['net_pnl'].sum(),
                    'avg_pnl_per_trade': trades_df['net_pnl'].mean(),
                    'best_trade': trades_df['net_pnl'].max(),
                    'worst_trade': trades_df['net_pnl'].min(),
                    'avg_hold_time': trades_df['bars_held'].mean() / 4,  # Convert to hours
                    'total_return_pct': trades_df['net_pnl'].sum() / INITIAL_CAPITAL * 100,
                    'avg_slippage': trades_df['slippage_cost'].mean(),
                    'total_commission': trades_df['commission_cost'].sum()
                }

                all_results.append(stats)
                all_trades.extend(trades)
                successful_symbols += 1

                print(f"  ✓ {stats['total_trades']} trades, {stats['win_rate']:.1f}% win rate, "
                      f"₹{stats['total_pnl']:.0f} PnL, {stats['total_return_pct']:.2f}% return")
            else:
                print(f"  ✗ Insufficient trades: {len(trades) if trades else 0}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    # Generate final report
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('total_pnl', ascending=False)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"ultimate_daily_backtest_{timestamp}.csv"
        results_df.to_csv(results_filename, index=False)

        trades_df = pd.DataFrame(all_trades)
        trades_filename = f"ultimate_daily_trades_{timestamp}.csv"
        trades_df.to_csv(trades_filename, index=False)

        print(f"\n" + "=" * 65)
        print("🏆 ULTIMATE DAILY TRADING RESULTS")
        print("=" * 65)
        print(f"Successfully processed: {successful_symbols}/{len(symbols)} symbols")
        print(f"Results saved to: {results_filename}")
        print(f"Trades saved to: {trades_filename}")
        print()

        # Display top performers
        print("TOP 10 PERFORMERS:")
        print("-" * 50)
        display_cols = ['symbol', 'total_trades', 'win_rate', 'total_pnl', 'total_return_pct', 'avg_hold_time']
        print(results_df[display_cols].head(10).round(2).to_string(index=False))

        # Overall statistics
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Trades: {results_df['total_trades'].sum()}")
        print(f"  Average Win Rate: {results_df['win_rate'].mean():.1f}%")
        print(f"  Total PnL: ₹{results_df['total_pnl'].sum():.0f}")
        print(f"  Average Return per Symbol: {results_df['total_return_pct'].mean():.2f}%")
        print(f"  Average Hold Time: {results_df['avg_hold_time'].mean():.1f} hours")
        print(f"  Profitable Symbols: {len(results_df[results_df['total_pnl'] > 0])}/{len(results_df)}")

        # Risk metrics
        if len(trades_df) > 0:
            print(f"\nRISK METRICS:")
            print(f"  Average Trade Size: ₹{(trades_df['shares'] * trades_df['entry_price']).mean():.0f}")
            print(f"  Average Slippage: ₹{trades_df['slippage_cost'].mean():.2f}")
            print(f"  Total Commission: ₹{trades_df['commission_cost'].sum():.0f}")

            # Calculate max drawdown
            trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
            running_max = trades_df['cumulative_pnl'].expanding().max()
            drawdown = trades_df['cumulative_pnl'] - running_max
            max_drawdown = drawdown.min()
            print(f"  Maximum Drawdown: ₹{max_drawdown:.0f}")

        return results_df, trades_df
    else:
        print("\nNo successful results. Check data and configuration.")
        return None, None

if __name__ == "__main__":
    results_df, trades_df = run_ultimate_daily_backtest()
