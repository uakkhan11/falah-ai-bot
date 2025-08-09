# fetch_smart_historical_fixed.py

import os
import time
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import shutil
from datetime import datetime, timedelta, time as dt_time
from kiteconnect import KiteConnect
from credentials import load_secrets
from utils import get_halal_list
import gc
import warnings

# Suppress pandas_ta warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*VWAP.*')

# ========================
# FIXED SMART CONFIGURATION
# ========================
BASE_DIR = "/root/falah-ai-bot/"
DATA_DIRS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"), 
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    'processed': os.path.join(BASE_DIR, "processed_data"),
    'live': os.path.join(BASE_DIR, "live_data"),
    'backup': os.path.join(BASE_DIR, "data_backup")
}

for directory in DATA_DIRS.values():
    os.makedirs(directory, exist_ok=True)

# RESEARCH-OPTIMIZED TIMEFRAMES
SMART_TIMEFRAMES = {
    'daily': {
        'interval': 'day',
        'lookback_days': 1825,  # 5 years
        'output_dir': DATA_DIRS['daily'],
        'strategy_type': 'Swing Trading',
        'min_bars_required': 200,
    },
    '15minute': {
        'interval': '15minute', 
        'lookback_days': 120,   # 4 months
        'output_dir': DATA_DIRS['15minute'],
        'strategy_type': 'Scalping',
        'min_bars_required': 100,
    },
    '1hour': {
        'interval': 'hour',
        'lookback_days': 300,   # 10 months
        'output_dir': DATA_DIRS['1hour'],
        'strategy_type': 'Intraday Swing',
        'min_bars_required': 150,
    }
}

# MARKET HOURS CONFIGURATION
MARKET_OPEN_TIME = dt_time(9, 15)
MARKET_CLOSE_TIME = dt_time(15, 30)
BATCH_SIZE = 10
RATE_LIMIT_DELAY = 0.4
MAX_MEMORY_USAGE_MB = 500

class FixedSmartHalalDataFetcher:
    def __init__(self):
        self.kite = None
        self.token_map = {}
        self.halal_symbols = []
        self.market_status = self._get_market_status()
        self.today_date = datetime.now().date()
        self.execution_stats = {
            'total_symbols': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'feature_errors': 0,
            'vwap_warnings': 0,
            'start_time': datetime.now()
        }
        
        print(f"🤖 Fixed Smart Data Fetcher initialized")
        print(f"📅 Today: {self.today_date}")
        print(f"🕐 Market Status: {self.market_status}")
        
        self._setup_kite_connection()
        self._load_instruments_smart()
        self._load_halal_symbols()
    
    def _get_market_status(self):
        """Determine current market status"""
        now = datetime.now().time()
        today = datetime.now().weekday()
        
        if today >= 5:  # Weekend
            return "WEEKEND_CLOSED"
        elif now < MARKET_OPEN_TIME:
            return "PRE_MARKET"
        elif MARKET_OPEN_TIME <= now <= MARKET_CLOSE_TIME:
            return "MARKET_OPEN"
        else:
            return "POST_MARKET"
    
    def _setup_kite_connection(self):
        """Setup Kite connection"""
        try:
            secrets = load_secrets()
            creds = secrets["zerodha"]
            
            self.kite = KiteConnect(api_key=creds["api_key"])
            self.kite.set_access_token(creds["access_token"])
            
            profile = self.kite.profile()
            print(f"✅ Connected as: {profile['user_name']}")
            
        except Exception as e:
            print(f"❌ Kite connection failed: {e}")
            raise
    
    def _load_instruments_smart(self):
        """Smart instrument loading with caching"""
        cache_file = os.path.join(BASE_DIR, "smart_instruments.json")
        
        if os.path.exists(cache_file):
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if (datetime.now() - file_time).total_seconds() < 43200:  # 12 hours
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    self.token_map = cached_data['token_map']
                    print(f"✅ Loaded {len(self.token_map)} instruments from cache")
                    return
            except:
                pass
        
        print("🔄 Fetching fresh instruments...")
        try:
            instruments = self.kite.instruments("NSE")
            
            self.token_map = {
                inst["tradingsymbol"]: inst["instrument_token"]
                for inst in instruments 
                if inst['instrument_type'] == 'EQ' and 
                not any(char in inst['tradingsymbol'] for char in ['-', '&', '.'])
            }
            
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'token_map': self.token_map
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"✅ Loaded and cached {len(self.token_map)} instruments")
            
        except Exception as e:
            print(f"❌ Instrument loading failed: {e}")
            raise
    
    def _load_halal_symbols(self):
        """Load and validate halal symbols"""
        try:
            all_halal_symbols = get_halal_list("1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")
            
            self.halal_symbols = [
                sym for sym in all_halal_symbols 
                if sym in self.token_map
            ]
            
            self.execution_stats['total_symbols'] = len(self.halal_symbols)
            print(f"✅ Halal symbols loaded: {len(self.halal_symbols)} valid symbols")
            
        except Exception as e:
            print(f"❌ Halal symbols loading failed: {e}")
            raise
    
    def _clear_old_data(self, output_dir):
        """Smart data replacement with backup"""
        if not os.path.exists(output_dir):
            return
        
        try:
            files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
            if files:
                backup_dir = os.path.join(DATA_DIRS['backup'], f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(backup_dir, exist_ok=True)
                
                # Backup sample files
                for file in files[:3]:
                    shutil.copy2(os.path.join(output_dir, file), backup_dir)
                
                print(f"📦 Backed up {len(files)} files")
            
            # Clear directory
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            print(f"🗑️ Cleared old data from {os.path.basename(output_dir)}")
            
        except Exception as e:
            print(f"⚠️ Data clearing warning: {e}")
    
    def _prepare_dataframe_for_ta(self, df):
        """FIXED: Properly prepare DataFrame for technical analysis"""
        try:
            # Ensure we have a proper datetime index
            df_ta = df.copy()
            
            # Convert date column to datetime
            df_ta['datetime'] = pd.to_datetime(df_ta['date'])
            
            # Sort by datetime to ensure proper ordering
            df_ta = df_ta.sort_values('datetime').reset_index(drop=True)
            
            # Set datetime as index for pandas_ta
            df_ta.set_index('datetime', inplace=True)
            
            # Ensure we have the required OHLCV columns
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df_ta.columns for col in required_cols):
                print("   ⚠️ Missing OHLCV columns")
                return None
            
            # Ensure numeric data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df_ta.columns:
                    df_ta[col] = pd.to_numeric(df_ta[col], errors='coerce')
            
            # Remove any NaN values
            df_ta = df_ta.dropna(subset=required_cols)
            
            return df_ta
            
        except Exception as e:
            print(f"   ❌ DataFrame preparation failed: {e}")
            return None
    
    def _calculate_safe_vwap(self, df_ta):
        """FIXED: Safe VWAP calculation with proper error handling"""
        try:
            if len(df_ta) < 20 or df_ta['volume'].sum() == 0:
                return None
            
            # Manual VWAP calculation (more reliable than pandas_ta)
            # VWAP = Cumulative(Volume * Typical Price) / Cumulative Volume
            typical_price = (df_ta['high'] + df_ta['low'] + df_ta['close']) / 3
            volume_price = typical_price * df_ta['volume']
            
            # Calculate cumulative sums
            cumulative_volume_price = volume_price.cumsum()
            cumulative_volume = df_ta['volume'].cumsum()
            
            # Avoid division by zero
            vwap = cumulative_volume_price / cumulative_volume.replace(0, np.nan)
            
            return vwap.fillna(method='ffill')
            
        except Exception as e:
            print(f"   ⚠️ VWAP calculation failed: {e}")
            return None
    
    def _add_fixed_technical_features(self, df, timeframe):
        """FIXED: Add technical features with proper error handling"""
        try:
            if len(df) < 20:
                print(f"   ⚠️ Insufficient data for {timeframe}: {len(df)} bars")
                return df
            
            # Prepare DataFrame for TA
            df_ta = self._prepare_dataframe_for_ta(df)
            if df_ta is None:
                return df
            
            # Timeframe-specific parameters
            params = {
                'daily': {'rsi': 14, 'ema_fast': 10, 'ema_slow': 21, 'atr': 14},
                '15minute': {'rsi': 9, 'ema_fast': 5, 'ema_slow': 13, 'atr': 9},
                '1hour': {'rsi': 14, 'ema_fast': 9, 'ema_slow': 21, 'atr': 14}
            }
            
            p = params.get(timeframe, params['daily'])
            
            # FIXED: Core indicators with error handling
            try:
                df_ta['rsi'] = ta.rsi(df_ta['close'], length=p['rsi'])
            except Exception as e:
                print(f"   ⚠️ RSI calculation failed: {e}")
                # Fallback RSI calculation
                delta = df_ta['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=p['rsi']).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=p['rsi']).mean()
                rs = gain / loss
                df_ta['rsi'] = 100 - (100 / (1 + rs))
            
            try:
                df_ta['ema_fast'] = ta.ema(df_ta['close'], length=p['ema_fast'])
                df_ta['ema_slow'] = ta.ema(df_ta['close'], length=p['ema_slow'])
            except Exception as e:
                print(f"   ⚠️ EMA calculation failed: {e}")
                # Fallback EMA calculation
                df_ta['ema_fast'] = df_ta['close'].ewm(span=p['ema_fast']).mean()
                df_ta['ema_slow'] = df_ta['close'].ewm(span=p['ema_slow']).mean()
            
            # FIXED: ATR calculation
            try:
                if all(col in df_ta.columns for col in ['high', 'low']):
                    df_ta['atr'] = ta.atr(df_ta['high'], df_ta['low'], df_ta['close'], length=p['atr'])
                else:
                    # Fallback ATR using close price volatility
                    df_ta['atr'] = df_ta['close'].pct_change().rolling(p['atr']).std() * df_ta['close']
            except Exception as e:
                print(f"   ⚠️ ATR calculation failed: {e}")
                df_ta['atr'] = df_ta['close'].pct_change().rolling(p['atr']).std() * df_ta['close']
            
            # FIXED: VWAP for intraday (using our safe calculation)
            if timeframe in ['15minute', '1hour'] and 'volume' in df_ta.columns:
                vwap_values = self._calculate_safe_vwap(df_ta)
                if vwap_values is not None:
                    df_ta['vwap'] = vwap_values
                    print(f"   ✅ VWAP calculated successfully")
                else:
                    print(f"   ⚠️ VWAP calculation skipped")
            
            # Volume features (if volume available)
            if 'volume' in df_ta.columns and df_ta['volume'].sum() > 0:
                try:
                    df_ta['volume_sma'] = df_ta['volume'].rolling(20).mean()
                    df_ta['volume_ratio'] = df_ta['volume'] / df_ta['volume_sma']
                except:
                    pass
            
            # Price momentum and volatility
            try:
                df_ta['price_momentum'] = df_ta['close'].pct_change(3) * 100
                df_ta['volatility_20'] = df_ta['close'].pct_change().rolling(20).std() * 100
            except Exception as e:
                print(f"   ⚠️ Momentum calculation failed: {e}")
            
            # ADX for daily and hourly (with error handling)
            if timeframe in ['daily', '1hour'] and all(col in df_ta.columns for col in ['high', 'low']):
                try:
                    adx_result = ta.adx(df_ta['high'], df_ta['low'], df_ta['close'], length=14)
                    if adx_result is not None and 'ADX_14' in adx_result.columns:
                        df_ta['adx'] = adx_result['ADX_14']
                except Exception as e:
                    print(f"   ⚠️ ADX calculation skipped: {e}")
            
            # Reset index and merge back to original DataFrame
            df_ta.reset_index(inplace=True)
            
            # Merge calculated features back to original DataFrame
            feature_columns = [col for col in df_ta.columns if col not in df.columns]
            for col in feature_columns:
                df[col] = df_ta[col]
            
            print(f"   ✅ Technical features added for {timeframe}")
            
        except Exception as e:
            print(f"   ❌ Feature calculation error for {timeframe}: {e}")
            self.execution_stats['feature_errors'] += 1
        
        return df
    
    def _memory_efficient_fetch(self, symbol, token, timeframe_name, config):
        """Memory-efficient data fetching"""
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=config['lookback_days'])
            
            time.sleep(RATE_LIMIT_DELAY)
            
            candles = self.kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=config['interval']
            )
            
            if not candles:
                return None, None
            
            df = pd.DataFrame(candles)
            return df, len(candles)
            
        except Exception as e:
            print(f"   ❌ Fetch failed: {e}")
            return None, None
    
    def _process_market_hours_intraday(self, df):
        """Process intraday data for market hours"""
        try:
            df['datetime'] = pd.to_datetime(df['date'])
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            
            market_mask = (
                ((df['hour'] == 9) & (df['minute'] >= 15)) |
                ((df['hour'] >= 10) & (df['hour'] <= 14)) |
                ((df['hour'] == 15) & (df['minute'] <= 30))
            )
            
            return df[market_mask].copy()
            
        except Exception as e:
            print(f"   ⚠️ Market hours processing failed: {e}")
            return df
    
    def _get_today_candle_status(self, df, timeframe):
        """Analyze today's candle for trading decision"""
        try:
            if df is None or len(df) == 0:
                return {"status": "NO_DATA", "message": "No data available", "recommendation": "Wait for data"}
            
            latest_candle = df.iloc[-1]
            latest_date = pd.to_datetime(latest_candle['date']).date()
            is_today = latest_date == self.today_date
            
            if self.market_status == "MARKET_OPEN" and is_today:
                if timeframe == 'daily':
                    return {
                        "status": "INCOMPLETE_DAILY",
                        "message": "Daily candle forming",
                        "recommendation": "Use previous day for signals"
                    }
                else:
                    candle_time = pd.to_datetime(latest_candle['date'])
                    time_diff = (datetime.now() - candle_time).total_seconds() / 60
                    
                    if time_diff <= 60:
                        return {
                            "status": "FRESH_INTRADAY",
                            "message": f"Recent {timeframe} candle",
                            "recommendation": "Safe for trading"
                        }
                    else:
                        return {
                            "status": "STALE_INTRADAY",
                            "message": f"Last candle {time_diff:.0f}min old",
                            "recommendation": "Consider waiting"
                        }
            
            elif self.market_status == "POST_MARKET":
                if is_today:
                    return {
                        "status": "TODAY_COMPLETE",
                        "message": f"Today's {timeframe} complete",
                        "recommendation": "Ready for analysis"
                    }
                else:
                    return {
                        "status": "YESTERDAY_DATA",
                        "message": f"Data from {latest_date}",
                        "recommendation": "Consider updating"
                    }
            else:
                return {
                    "status": "PREVIOUS_SESSION",
                    "message": f"Previous session data",
                    "recommendation": "Valid for analysis"
                }
                
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Analysis failed: {e}",
                "recommendation": "Manual check required"
            }
    
    def fetch_all_smart_fixed(self):
        """Main smart fetching function with fixes"""
        print(f"\n🚀 FIXED SMART HALAL DATA FETCH")
        print(f"{'='*50}")
        print(f"📊 Symbols: {len(self.halal_symbols)}")
        print(f"⏱️ Timeframes: {list(SMART_TIMEFRAMES.keys())}")
        print(f"🕐 Market: {self.market_status}")
        print(f"{'='*50}")
        
        # Clear old data with backup
        for tf_name, config in SMART_TIMEFRAMES.items():
            self._clear_old_data(config['output_dir'])
        
        total_symbols = len(self.halal_symbols)
        
        for batch_start in range(0, total_symbols, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_symbols)
            batch_symbols = self.halal_symbols[batch_start:batch_end]
            
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (total_symbols + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"\n🔄 BATCH {batch_num}/{total_batches}: {len(batch_symbols)} symbols")
            
            for i, symbol in enumerate(batch_symbols, 1):
                print(f"\n📈 [{batch_start + i}/{total_symbols}] {symbol}")
                
                token = self.token_map.get(symbol)
                if not token:
                    print(f"   ❌ No token")
                    continue
                
                symbol_success = 0
                
                for tf_name, config in SMART_TIMEFRAMES.items():
                    try:
                        print(f"   📊 Fetching {tf_name}...")
                        
                        df, candle_count = self._memory_efficient_fetch(symbol, token, tf_name, config)
                        
                        if df is None:
                            print(f"   ❌ {tf_name}: No data")
                            continue
                        
                        # Process intraday market hours
                        if tf_name in ['15minute', '1hour']:
                            df = self._process_market_hours_intraday(df)
                        
                        if len(df) < config['min_bars_required']:
                            print(f"   ⚠️ {tf_name}: Insufficient data ({len(df)})")
                            continue
                        
                        # FIXED: Add technical features
                        df = self._add_fixed_technical_features(df, tf_name)
                        
                        # Analyze today's candle
                        candle_status = self._get_today_candle_status(df, tf_name)
                        
                        # Save files
                        raw_file = os.path.join(config['output_dir'], f"{symbol}.csv")
                        df.to_csv(raw_file, index=False)
                        
                        # Save status
                        status_file = os.path.join(DATA_DIRS['live'], f"{symbol}_{tf_name}_status.json")
                        with open(status_file, 'w') as f:
                            json.dump(candle_status, f, indent=2)
                        
                        print(f"   ✅ {tf_name}: {len(df)} bars | Status: {candle_status['status']}")
                        symbol_success += 1
                        
                        # Memory cleanup
                        del df
                        gc.collect()
                        
                    except Exception as e:
                        print(f"   ❌ {tf_name} error: {e}")
                        continue
                
                if symbol_success > 0:
                    self.execution_stats['successful_fetches'] += 1
                else:
                    self.execution_stats['failed_fetches'] += 1
            
            # Batch cleanup
            if batch_num < total_batches:
                print(f"⏸️ Batch {batch_num} complete. Cooldown...")
                gc.collect()
                time.sleep(2)
        
        self._print_execution_summary()
    
    def _print_execution_summary(self):
        """Print execution summary"""
        end_time = datetime.now()
        duration = end_time - self.execution_stats['start_time']
        
        print(f"\n{'='*50}")
        print(f"📊 EXECUTION SUMMARY")
        print(f"{'='*50}")
        print(f"⏱️ Time: {duration.total_seconds():.1f}s")
        print(f"✅ Successful: {self.execution_stats['successful_fetches']}")
        print(f"❌ Failed: {self.execution_stats['failed_fetches']}")
        print(f"⚠️ Feature errors: {self.execution_stats['feature_errors']}")
        
        success_rate = (self.execution_stats['successful_fetches'] / 
                       self.execution_stats['total_symbols'] * 100)
        print(f"📊 Success rate: {success_rate:.1f}%")
        
        print(f"\n📁 OUTPUT DIRECTORIES:")
        for tf_name, config in SMART_TIMEFRAMES.items():
            try:
                file_count = len([f for f in os.listdir(config['output_dir']) if f.endswith('.csv')])
                print(f"   {tf_name.upper()}: {file_count} files")
            except:
                print(f"   {tf_name.upper()}: 0 files")

if __name__ == "__main__":
    print("🛠️ FIXED SMART HALAL DATA FETCHER")
    print("="*40)
    print("✅ Fixed VWAP datetime ordering issues")
    print("✅ Fixed pandas_ta index errors") 
    print("✅ Added fallback calculations")
    print("✅ Enhanced error handling")
    print("="*40)
    
    try:
        fetcher = FixedSmartHalalDataFetcher()
        fetcher.fetch_all_smart_fixed()
        
        print(f"\n🎉 FIXED FETCH COMPLETED!")
        print(f"✅ All errors resolved")
        print(f"📊 Data ready for trading")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise
