# fetch_intraday_data_robust.py

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import pandas_ta as ta
import time
import sqlite3
from credentials import load_secrets
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
INTRADAY_DIR = "/root/falah-ai-bot/intraday_data/"
PROCESSED_DIR = "/root/falah-ai-bot/processed_intraday/"
DB_PATH = "/root/falah-ai-bot/intraday_cache.db"
TOKEN_MAP_FILE = "/root/falah-ai-bot/symbol_to_token.json"
INSTRUMENT_MAP_FILE = "/root/falah-ai-bot/instrument_token_map.json"

# Enhanced configuration
TIMEFRAMES = ["15minute", "hour"]
DAYS_BACK = 30
MAX_SYMBOLS_PER_BATCH = 20  # Reduced for API limits
REQUEST_DELAY = 0.4  # 400ms delay (2.5 req/sec, safer than 3)
MAX_RETRIES = 3
CHUNK_SIZE = 5  # Process in smaller chunks

class RobustKiteDataFetcher:
    def __init__(self):
        self.kite = None
        self.token_map = {}
        self.reverse_token_map = {}
        self.db_connection = None
        self.failed_requests = []
        self.successful_requests = 0
        self.rate_limit_hits = 0
        
        self._setup_database()
        self._load_token_maps()
        self._setup_kite_connection()
    
    def _setup_database(self):
        """Setup SQLite cache database"""
        try:
            self.db_connection = sqlite3.connect(DB_PATH)
            cursor = self.db_connection.cursor()
            
            # Create cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_cache (
                    symbol TEXT,
                    timeframe TEXT,
                    date_key TEXT,
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, timeframe, date_key)
                )
            ''')
            
            # Create metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fetch_metadata (
                    symbol TEXT,
                    timeframe TEXT,
                    last_fetch TIMESTAMP,
                    status TEXT,
                    error_count INTEGER DEFAULT 0,
                    PRIMARY KEY (symbol, timeframe)
                )
            ''')
            
            self.db_connection.commit()
            print("✅ Database cache initialized")
            
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            self.db_connection = None
    
    def _load_token_maps(self):
        """Load pre-generated token maps"""
        try:
            if os.path.exists(TOKEN_MAP_FILE):
                with open(TOKEN_MAP_FILE, 'r') as f:
                    self.token_map = json.load(f)
                print(f"✅ Loaded {len(self.token_map)} symbol tokens")
            
            if os.path.exists(INSTRUMENT_MAP_FILE):
                with open(INSTRUMENT_MAP_FILE, 'r') as f:
                    self.reverse_token_map = json.load(f)
                print(f"✅ Loaded reverse token map")
                
        except Exception as e:
            print(f"⚠️ Token map loading failed: {e}")
            self._generate_token_maps()
    
    def _generate_token_maps(self):
        """Fallback: Generate token maps if files don't exist"""
        print("🔄 Generating token maps...")
        try:
            if not self.kite:
                self._setup_kite_connection()
            
            instruments = self.kite.instruments("NSE")
            
            # Create both mappings
            self.token_map = {}
            self.reverse_token_map = {}
            
            for inst in instruments:
                if inst['instrument_type'] == 'EQ':  # Only equity stocks
                    symbol = inst['tradingsymbol']
                    token = inst['instrument_token']
                    
                    self.token_map[symbol] = token
                    self.reverse_token_map[str(token)] = symbol
            
            # Save to files
            with open(TOKEN_MAP_FILE, 'w') as f:
                json.dump(self.token_map, f)
            
            with open(INSTRUMENT_MAP_FILE, 'w') as f:
                json.dump(self.reverse_token_map, f)
            
            print(f"✅ Generated token maps: {len(self.token_map)} symbols")
            
        except Exception as e:
            print(f"❌ Token map generation failed: {e}")
    
    def _setup_kite_connection(self):
        """Setup robust Kite connection with retry logic"""
        try:
            secrets = load_secrets()
            creds = secrets["zerodha"]
            
            self.kite = KiteConnect(api_key=creds["api_key"])
            self.kite.set_access_token(creds["access_token"])
            
            # Test connection
            profile = self.kite.profile()
            print(f"✅ Connected as: {profile['user_name']}")
            
            # Setup session with retry strategy
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
        except Exception as e:
            print(f"❌ Kite connection failed: {e}")
            raise
    
    def _check_cache(self, symbol, timeframe, from_date, to_date):
        """Check if data exists in cache"""
        if not self.db_connection:
            return None
        
        try:
            cursor = self.db_connection.cursor()
            date_key = f"{from_date.strftime('%Y%m%d')}_{to_date.strftime('%Y%m%d')}"
            
            cursor.execute('''
                SELECT data FROM price_cache 
                WHERE symbol = ? AND timeframe = ? AND date_key = ?
            ''', (symbol, timeframe, date_key))
            
            result = cursor.fetchone()
            if result:
                cached_data = json.loads(result[0])
                print(f"💾 Using cached data for {symbol}")
                return pd.DataFrame(cached_data)
            
        except Exception as e:
            print(f"⚠️ Cache check failed: {e}")
        
        return None
    
    def _save_to_cache(self, symbol, timeframe, from_date, to_date, df):
        """Save data to cache"""
        if not self.db_connection:
            return
        
        try:
            cursor = self.db_connection.cursor()
            date_key = f"{from_date.strftime('%Y%m%d')}_{to_date.strftime('%Y%m%d')}"
            data_json = df.to_json(orient='records')
            
            cursor.execute('''
                INSERT OR REPLACE INTO price_cache (symbol, timeframe, date_key, data)
                VALUES (?, ?, ?, ?)
            ''', (symbol, timeframe, date_key, data_json))
            
            # Update metadata
            cursor.execute('''
                INSERT OR REPLACE INTO fetch_metadata (symbol, timeframe, last_fetch, status)
                VALUES (?, ?, ?, 'success')
            ''', (symbol, timeframe, datetime.now()))
            
            self.db_connection.commit()
            
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")
    
    def _fetch_with_retry(self, symbol, timeframe, from_date, to_date, max_retries=MAX_RETRIES):
        """Fetch data with exponential backoff retry"""
        token = self.token_map.get(symbol)
        if not token:
            print(f"⚠️ No token for {symbol}")
            return None
        
        for attempt in range(max_retries):
            try:
                print(f"📡 Fetching {symbol} ({timeframe}) - Attempt {attempt + 1}")
                
                # Add jitter to prevent thundering herd
                jitter = 0.1 + (attempt * 0.2)
                time.sleep(REQUEST_DELAY + jitter)
                
                candles = self.kite.historical_data(
                    instrument_token=token,
                    from_date=from_date,
                    to_date=to_date,
                    interval=timeframe
                )
                
                if candles:
                    self.successful_requests += 1
                    return pd.DataFrame(candles)
                else:
                    print(f"⚠️ No data returned for {symbol}")
                    return None
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if "rate limit" in error_msg or "429" in error_msg:
                    self.rate_limit_hits += 1
                    wait_time = min(60, (2 ** attempt) + jitter)  # Cap at 60 seconds
                    print(f"⏱️ Rate limit hit. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                elif "network" in error_msg or "timeout" in error_msg:
                    wait_time = (2 ** attempt) + jitter
                    print(f"🌐 Network error. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    print(f"❌ Fetch failed for {symbol}: {e}")
                    self.failed_requests.append(f"{symbol}_{timeframe}_{e}")
                    break
        
        return None
    
    def fetch_intraday_data_robust(self, symbols, timeframes=TIMEFRAMES, days=DAYS_BACK):
        """Main robust fetching function"""
        print(f"🚀 Starting robust intraday data fetch for {len(symbols)} symbols")
        
        # Create directories
        for directory in [INTRADAY_DIR, PROCESSED_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        successful_symbols = []
        failed_symbols = []
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Process in chunks to avoid overwhelming API
        symbol_chunks = [symbols[i:i + CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]
        
        for chunk_idx, symbol_chunk in enumerate(symbol_chunks):
            print(f"\n📦 Processing chunk {chunk_idx + 1}/{len(symbol_chunks)}")
            
            for timeframe in timeframes:
                print(f"\n⏱️ Timeframe: {timeframe}")
                
                # Create timeframe directories
                timeframe_dir = os.path.join(INTRADAY_DIR, timeframe)
                processed_timeframe_dir = os.path.join(PROCESSED_DIR, timeframe)
                
                for directory in [timeframe_dir, processed_timeframe_dir]:
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                
                for symbol in symbol_chunk:
                    try:
                        # Check cache first
                        cached_df = self._check_cache(symbol, timeframe, from_date, to_date)
                        
                        if cached_df is not None and len(cached_df) > 50:
                            df = cached_df
                            print(f"💾 {symbol}: Using cached data ({len(df)} bars)")
                        else:
                            # Fetch from API
                            df = self._fetch_with_retry(symbol, timeframe, from_date, to_date)
                            
                            if df is None or len(df) == 0:
                                failed_symbols.append(f"{symbol}_{timeframe}")
                                continue
                            
                            # Save to cache
                            self._save_to_cache(symbol, timeframe, from_date, to_date, df)
                        
                        # Process data
                        df = self._process_intraday_data(df, symbol, timeframe)
                        
                        if df is None or len(df) < 50:
                            failed_symbols.append(f"{symbol}_{timeframe}_insufficient")
                            continue
                        
                        # Save files
                        raw_file = os.path.join(timeframe_dir, f"{symbol}.csv")
                        processed_file = os.path.join(processed_timeframe_dir, f"{symbol}.csv")
                        
                        df.to_csv(raw_file, index=False)
                        
                        # Add technical features
                        df_processed = self._calculate_intraday_features(df, timeframe)
                        df_processed.to_csv(processed_file, index=False)
                        
                        successful_symbols.append(f"{symbol}_{timeframe}")
                        print(f"✅ {symbol}: {len(df)} bars processed")
                        
                    except Exception as e:
                        print(f"❌ {symbol} failed: {e}")
                        failed_symbols.append(f"{symbol}_{timeframe}_error")
                        continue
            
            # Longer pause between chunks
            if chunk_idx < len(symbol_chunks) - 1:
                print("⏸️ Chunk complete. Cooling down...")
                time.sleep(2)
        
        self._print_summary(successful_symbols, failed_symbols)
        return successful_symbols, failed_symbols
    
    def _process_intraday_data(self, df, symbol, timeframe):
        """Process and validate intraday data"""
        try:
            # Add datetime features
            df['datetime'] = pd.to_datetime(df['date'])
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            df['day_of_week'] = df['datetime'].dt.dayofweek
            
            # Filter market hours (9:15 AM to 3:30 PM)
            market_mask = (
                ((df['hour'] == 9) & (df['minute'] >= 15)) |  # 9:15 AM onwards
                ((df['hour'] >= 10) & (df['hour'] <= 14)) |   # 10 AM to 2:59 PM
                ((df['hour'] == 15) & (df['minute'] <= 30))   # Up to 3:30 PM
            )
            
            df = df[market_mask].copy()
            
            # Data quality checks
            if len(df) == 0:
                print(f"⚠️ No market hours data for {symbol}")
                return None
            
            # Remove obvious data errors
            required_cols = ['open', 'high', 'low', 'close']
            if all(col in df.columns for col in required_cols):
                # Basic OHLC validation
                valid_mask = (
                    (df['high'] >= df[['open', 'close', 'low']].max(axis=1)) &
                    (df['low'] <= df[['open', 'close', 'high']].min(axis=1)) &
                    (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)
                )
                df = df[valid_mask]
            
            return df
            
        except Exception as e:
            print(f"❌ Data processing failed for {symbol}: {e}")
            return None
    
    def _calculate_intraday_features(self, df, timeframe):
        """Calculate optimized intraday features"""
        try:
            # Timeframe-specific parameters
            if timeframe == "15minute":
                fast_period = 7
                slow_period = 21
                rsi_period = 9
            else:  # hour
                fast_period = 9
                slow_period = 34
                rsi_period = 14
            
            # Core indicators
            df['rsi'] = ta.rsi(df['close'], length=rsi_period)
            df['ema_fast'] = ta.ema(df['close'], length=fast_period)
            df['ema_slow'] = ta.ema(df['close'], length=slow_period)
            
            if all(col in df.columns for col in ['high', 'low', 'volume']):
                df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=rsi_period)
                df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
                df['stoch'] = ta.stoch(df['high'], df['low'], df['close'])['STOCHk_14_3_3']
            
            # Intraday-specific features
            df['price_momentum'] = df['close'].pct_change(periods=3) * 100
            df['volatility'] = df['close'].pct_change().rolling(20).std() * 100
            
            return df
            
        except Exception as e:
            print(f"⚠️ Feature calculation failed: {e}")
            return df
    
    def _print_summary(self, successful, failed):
        """Print comprehensive summary"""
        print(f"\n{'='*60}")
        print("📊 ROBUST FETCHING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"🌐 API Requests: {self.successful_requests}")
        print(f"⚠️ Rate Limits Hit: {self.rate_limit_hits}")
        
        if self.failed_requests:
            print(f"\n❌ Failed Requests Sample:")
            for req in self.failed_requests[:5]:
                print(f"   - {req}")
    
    def close(self):
        """Clean up resources"""
        if self.db_connection:
            self.db_connection.close()

# Convenience function
def fetch_intraday_data_with_resilience(symbols, timeframes=["15minute", "hour"], days=30):
    """Main entry point for robust data fetching"""
    fetcher = RobustKiteDataFetcher()
    
    try:
        successful, failed = fetcher.fetch_intraday_data_robust(symbols, timeframes, days)
        return successful, failed
    finally:
        fetcher.close()

if __name__ == "__main__":
    print("🛡️ ROBUST INTRADAY DATA FETCHER")
    print("="*50)
    
    # Load symbols
    SCREENED_FILE = "/root/falah-ai-bot/final_screened.json"
    symbols = []
    
    if os.path.exists(SCREENED_FILE):
        with open(SCREENED_FILE) as f:
            data = json.load(f)
        if isinstance(data, dict):
            symbols = list(data.keys())[:MAX_SYMBOLS_PER_BATCH]  # Limit symbols
        elif isinstance(data, list):
            symbols = data[:MAX_SYMBOLS_PER_BATCH]
    else:
        # Fallback liquid stocks
        symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'SBIN', 
                  'BAJFINANCE', 'HDFCBANK', 'HINDUNILVR', 'ITC']
    
    print(f"🎯 Fetching data for {len(symbols)} symbols")
    
    # Run robust fetching
    successful, failed = fetch_intraday_data_with_resilience(symbols)
    
    print(f"\n🎊 COMPLETED!")
    print(f"Check directories: {INTRADAY_DIR} and {PROCESSED_DIR}")
