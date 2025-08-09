# fetch_unified_historical_intraday.py

import os
import time
import pandas as pd
import pandas_ta as ta
import json
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from credentials import load_secrets
from utils import get_halal_list

# Configuration
BASE_DIR = "/root/falah-ai-bot/"
DAILY_OUTPUT_DIR = os.path.join(BASE_DIR, "historical_data")
INTRADAY_OUTPUT_DIR = os.path.join(BASE_DIR, "intraday_data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

# Create all directories
for directory in [DAILY_OUTPUT_DIR, INTRADAY_OUTPUT_DIR, PROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Fetching configuration
BATCH_SIZE = 15  # Reduced for intraday API limits
RATE_LIMIT_DELAY = 0.4  # 400ms between requests
DAILY_LOOKBACK_YEARS = 5  # Changed from 3 to 5 years
INTRADAY_LOOKBACK_DAYS = 60  # Increased from 30 days

# Timeframes to fetch
TIMEFRAMES = {
    'daily': {
        'interval': 'day',
        'lookback': timedelta(days=DAILY_LOOKBACK_YEARS * 365),
        'output_dir': DAILY_OUTPUT_DIR,
        'description': 'Daily/Swing Trading Data'
    },
    '15minute': {
        'interval': '15minute', 
        'lookback': timedelta(days=INTRADAY_LOOKBACK_DAYS),
        'output_dir': os.path.join(INTRADAY_OUTPUT_DIR, '15minute'),
        'description': '15-Minute Intraday Data'
    },
    '1hour': {
        'interval': 'hour',
        'lookback': timedelta(days=INTRADAY_LOOKBACK_DAYS * 2),  # More days for hourly
        'output_dir': os.path.join(INTRADAY_OUTPUT_DIR, '1hour'),
        'description': '1-Hour Intraday Data'
    }
}

class UnifiedHalalDataFetcher:
    def __init__(self):
        self.kite = None
        self.token_map = {}
        self.halal_symbols = []
        self.fetch_summary = {
            'total_symbols': 0,
            'successful_symbols': {},
            'failed_symbols': {},
            'timeframe_stats': {}
        }
        
        self._setup_kite_connection()
        self._load_instruments()
        self._load_halal_symbols()
    
    def _setup_kite_connection(self):
        """Setup Kite connection with error handling"""
        try:
            secrets = load_secrets()
            creds = secrets["zerodha"]
            
            self.kite = KiteConnect(api_key=creds["api_key"])
            self.kite.set_access_token(creds["access_token"])
            
            # Test connection
            profile = self.kite.profile()
            print(f"✅ Connected to Zerodha as: {profile['user_name']}")
            
        except Exception as e:
            print(f"❌ Kite connection failed: {e}")
            raise
    
    def _load_instruments(self):
        """Load instrument tokens with caching"""
        cache_file = os.path.join(BASE_DIR, "instrument_cache.json")
        
        # Try to load from cache first (faster)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is less than 1 day old
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2020-01-01'))
                if (datetime.now() - cache_time).days < 1:
                    self.token_map = cache_data['token_map']
                    print(f"✅ Loaded {len(self.token_map)} instruments from cache")
                    return
            except:
                pass
        
        # Fetch fresh data from API
        print("📥 Downloading fresh instrument list from Zerodha...")
        try:
            instruments = self.kite.instruments("NSE")
            
            # Filter for equity instruments only
            equity_instruments = [
                inst for inst in instruments 
                if inst['instrument_type'] == 'EQ'
            ]
            
            self.token_map = {
                inst["tradingsymbol"]: inst["instrument_token"]
                for inst in equity_instruments
            }
            
            # Cache the data
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'token_map': self.token_map
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"✅ Loaded and cached {len(self.token_map)} NSE equity instruments")
            
        except Exception as e:
            print(f"❌ Failed to load instruments: {e}")
            raise
    
    def _load_halal_symbols(self):
        """Load halal symbols list"""
        try:
            self.halal_symbols = get_halal_list("1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")
            self.fetch_summary['total_symbols'] = len(self.halal_symbols)
            print(f"✅ Loaded {len(self.halal_symbols)} halal symbols")
            
            # Filter symbols that have tokens
            valid_symbols = [sym for sym in self.halal_symbols if sym in self.token_map]
            missing_symbols = set(self.halal_symbols) - set(valid_symbols)
            
            if missing_symbols:
                print(f"⚠️ {len(missing_symbols)} symbols not found in NSE: {list(missing_symbols)[:5]}...")
            
            self.halal_symbols = valid_symbols
            print(f"📊 {len(self.halal_symbols)} symbols have valid tokens and will be fetched")
            
        except Exception as e:
            print(f"❌ Failed to load halal symbols: {e}")
            raise
    
    def _process_market_hours_data(self, df, timeframe):
        """Filter and process data for market hours"""
        if timeframe == 'daily':
            return df  # No filtering needed for daily data
        
        try:
            # Add datetime column if not exists
            if 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
            
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            
            # Filter for market hours (9:15 AM to 3:30 PM)
            market_hours_mask = (
                ((df['hour'] == 9) & (df['minute'] >= 15)) |  # From 9:15 AM
                ((df['hour'] >= 10) & (df['hour'] <= 14)) |   # 10 AM to 2:59 PM
                ((df['hour'] == 15) & (df['minute'] <= 30))   # Until 3:30 PM
            )
            
            filtered_df = df[market_hours_mask].copy()
            print(f"   Filtered to {len(filtered_df)} market hours bars (from {len(df)} total)")
            
            return filtered_df
            
        except Exception as e:
            print(f"   ⚠️ Market hours filtering failed: {e}")
            return df
    
    def _add_technical_features(self, df, timeframe):
        """Add timeframe-specific technical indicators"""
        try:
            if len(df) < 20:  # Need minimum data for indicators
                return df
            
            # Timeframe-specific parameters
            if timeframe == 'daily':
                rsi_period = 14
                ema_short = 10
                ema_long = 21
                atr_period = 14
            elif timeframe == '15minute':
                rsi_period = 9
                ema_short = 5
                ema_long = 13
                atr_period = 9
            else:  # 1hour
                rsi_period = 14
                ema_short = 9
                ema_long = 21
                atr_period = 14
            
            # Core indicators
            df['rsi'] = ta.rsi(df['close'], length=rsi_period)
            df['ema_short'] = ta.ema(df['close'], length=ema_short)
            df['ema_long'] = ta.ema(df['close'], length=ema_long)
            
            # Volatility and volume indicators (if OHLCV available)
            if all(col in df.columns for col in ['high', 'low', 'volume']):
                df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
                
                # Volume indicators (if volume data exists)
                if df['volume'].sum() > 0:
                    df['volume_ma'] = df['volume'].rolling(20).mean()
                    df['volume_ratio'] = df['volume'] / df['volume_ma']
                    
                    # VWAP for intraday
                    if timeframe in ['15minute', '1hour']:
                        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            
            # Additional indicators
            df['price_change'] = df['close'].pct_change() * 100
            df['volatility'] = df['price_change'].rolling(20).std()
            
            # ADX for trend strength (daily and hourly)
            if timeframe in ['daily', '1hour'] and all(col in df.columns for col in ['high', 'low']):
                df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
            
            print(f"   Added technical features for {timeframe}")
            
        except Exception as e:
            print(f"   ⚠️ Technical features calculation failed: {e}")
        
        return df
    
    def _fetch_timeframe_data(self, symbol, token, timeframe_config, timeframe_name):
        """Fetch data for a specific timeframe"""
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=DAILY_LOOKBACK_YEARS * 365)
            
            print(f"📅 Optimized backtest period: {from_date.date()} to {to_date.date()}")
            print(f"📊 Expected trades: ~400-600 (vs ~200-300 with 3 years)")
            
            # Add rate limiting delay
            time.sleep(RATE_LIMIT_DELAY)
            
            # Fetch data from Zerodha
            candles = self.kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=timeframe_config['interval']
            )
            
            if not candles:
                print(f"   ⚠️ No {timeframe_name} data received")
                return None, None
            
            # Convert to DataFrame
            raw_df = pd.DataFrame(candles)
            print(f"   ✅ Received {len(raw_df)} {timeframe_name} candles")
            
            # Process market hours (for intraday)
            filtered_df = self._process_market_hours_data(raw_df, timeframe_name)
            
            if len(filtered_df) == 0:
                print(f"   ❌ No valid data after market hours filtering")
                return None, None
            
            # Add technical features
            processed_df = self._add_technical_features(filtered_df, timeframe_name)
            
            return raw_df, processed_df
            
        except Exception as e:
            print(f"   ❌ {timeframe_name} fetch failed: {e}")
            return None, None
    
    def fetch_all_timeframes(self):
        """Fetch data for all timeframes and symbols"""
        print(f"🚀 STARTING UNIFIED HALAL DATA FETCH")
        print(f"{'='*60}")
        print(f"📊 Symbols: {len(self.halal_symbols)}")
        print(f"⏱️ Timeframes: {list(TIMEFRAMES.keys())}")
        print(f"📅 Daily lookback: {DAILY_LOOKBACK_YEARS} years")
        print(f"📅 Intraday lookback: {INTRADAY_LOOKBACK_DAYS} days")
        print(f"{'='*60}")
        
        # Create all output directories
        for tf_config in TIMEFRAMES.values():
            os.makedirs(tf_config['output_dir'], exist_ok=True)
        
        # Process symbols in batches
        total_symbols = len(self.halal_symbols)
        
        for batch_start in range(0, total_symbols, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_symbols)
            batch_symbols = self.halal_symbols[batch_start:batch_end]
            
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (total_symbols + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"\n🔄 BATCH {batch_num}/{total_batches}: Processing {len(batch_symbols)} symbols")
            print(f"   Symbols: {', '.join(batch_symbols)}")
            
            for i, symbol in enumerate(batch_symbols, 1):
                print(f"\n📈 [{batch_start + i}/{total_symbols}] Processing {symbol}")
                
                # Get token
                token = self.token_map.get(symbol)
                if not token:
                    print(f"   ❌ No token found for {symbol}")
                    continue
                
                symbol_success = {}
                symbol_failed = {}
                
                # Fetch each timeframe
                for tf_name, tf_config in TIMEFRAMES.items():
                    try:
                        raw_df, processed_df = self._fetch_timeframe_data(symbol, token, tf_config, tf_name)
                        
                        if raw_df is not None and processed_df is not None:
                            # Save raw data
                            raw_file = os.path.join(tf_config['output_dir'], f"{symbol}.csv")
                            raw_df.to_csv(raw_file, index=False)
                            
                            # Save processed data
                            processed_dir = os.path.join(PROCESSED_DIR, tf_name)
                            os.makedirs(processed_dir, exist_ok=True)
                            processed_file = os.path.join(processed_dir, f"{symbol}.csv")
                            processed_df.to_csv(processed_file, index=False)
                            
                            symbol_success[tf_name] = len(processed_df)
                            print(f"   ✅ {tf_name}: {len(processed_df)} bars saved")
                            
                        else:
                            symbol_failed[tf_name] = "No data received"
                            
                    except Exception as e:
                        symbol_failed[tf_name] = str(e)
                        print(f"   ❌ {tf_name} failed: {e}")
                
                # Update summary
                if symbol_success:
                    self.fetch_summary['successful_symbols'][symbol] = symbol_success
                if symbol_failed:
                    self.fetch_summary['failed_symbols'][symbol] = symbol_failed
            
            # Longer pause between batches
            if batch_num < total_batches:
                print(f"⏸️ Batch {batch_num} complete. Cooling down for 3 seconds...")
                time.sleep(3)
        
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print comprehensive summary of the fetch operation"""
        print(f"\n{'='*70}")
        print(f"📊 UNIFIED HALAL DATA FETCH SUMMARY")
        print(f"{'='*70}")
        
        total_symbols = len(self.halal_symbols)
        successful_symbols = len(self.fetch_summary['successful_symbols'])
        failed_symbols = len(self.fetch_summary['failed_symbols'])
        
        print(f"Total Symbols Processed: {total_symbols}")
        print(f"✅ Successful: {successful_symbols} ({successful_symbols/total_symbols*100:.1f}%)")
        print(f"❌ Failed: {failed_symbols} ({failed_symbols/total_symbols*100:.1f}%)")
        
        # Timeframe statistics
        print(f"\n📈 TIMEFRAME BREAKDOWN:")
        for tf_name in TIMEFRAMES.keys():
            tf_success = sum(1 for s in self.fetch_summary['successful_symbols'].values() if tf_name in s)
            print(f"   {tf_name.upper()}: {tf_success} symbols successful")
        
        # Directory information
        print(f"\n📁 OUTPUT DIRECTORIES:")
        for tf_name, tf_config in TIMEFRAMES.items():
            print(f"   {tf_name.upper()}: {tf_config['output_dir']}")
        print(f"   PROCESSED: {PROCESSED_DIR}")
        
        # Sample successful symbols
        if self.fetch_summary['successful_symbols']:
            sample_symbols = list(self.fetch_summary['successful_symbols'].keys())[:5]
            print(f"\n✅ SAMPLE SUCCESSFUL SYMBOLS:")
            for symbol in sample_symbols:
                print(f"   {symbol}: {list(self.fetch_summary['successful_symbols'][symbol].keys())}")
        
        # Failed symbols (if any)
        if self.fetch_summary['failed_symbols']:
            failed_sample = list(self.fetch_summary['failed_symbols'].keys())[:5]
            print(f"\n❌ SAMPLE FAILED SYMBOLS:")
            for symbol in failed_sample:
                print(f"   {symbol}: {list(self.fetch_summary['failed_symbols'][symbol].keys())}")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Check the output directories for your data files")
        print(f"2. Use daily data for swing trading backtests")
        print(f"3. Use intraday data for scalping backtests")
        print(f"4. Processed files include technical indicators")

def save_nifty_reference():
    """Save NIFTY index data as reference (enhanced)"""
    try:
        print("\n📊 Saving NIFTY reference data...")
        import yfinance as yf
        
        # Fetch NIFTY data
        nifty_df = yf.download("^NSEI", start="2020-01-01", end=datetime.now())
        nifty_df.reset_index(inplace=True)
        
        # Save to multiple directories for reference
        nifty_files = [
            os.path.join(DAILY_OUTPUT_DIR, "NIFTY.csv"),
            os.path.join(PROCESSED_DIR, "daily", "NIFTY.csv")
        ]
        
        for nifty_file in nifty_files:
            os.makedirs(os.path.dirname(nifty_file), exist_ok=True)
            nifty_df.to_csv(nifty_file, index=False)
        
        print(f"✅ NIFTY reference data saved ({len(nifty_df)} rows)")
        
    except Exception as e:
        print(f"⚠️ NIFTY data save failed: {e}")

if __name__ == "__main__":
    try:
        # Create and run the unified fetcher
        fetcher = UnifiedHalalDataFetcher()
        fetcher.fetch_all_timeframes()
        
        # Save NIFTY reference
        save_nifty_reference()
        
        print(f"\n🎉 UNIFIED HALAL DATA FETCH COMPLETED!")
        print(f"All your halal stocks now have daily and intraday data ready for backtesting!")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise
