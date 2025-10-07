#!/usr/bin/env python3
import os
import time
import json
import shutil
import logging
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta
from kiteconnect import KiteConnect
from utils import get_halal_list

# -------------------- CONFIG --------------------
BASE_DIR = "/root/falah-ai-bot"
TOKENS_FILE = os.path.join(BASE_DIR, "kite_tokens.json")  # Your existing token file

DATA_DIRS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '5minute': os.path.join(BASE_DIR, "five_minute_data"),
    'backup': os.path.join(BASE_DIR, "data_backup")
}

SMART_TIMEFRAMES = {
    'daily':    {'interval': 'day',       'lookback_days': 1825, 'output_dir': DATA_DIRS['daily'],     'min_bars_required': 200},
    '15minute': {'interval': '15minute',  'lookback_days': 120,  'output_dir': DATA_DIRS['15minute'], 'min_bars_required': 100},
    '1hour':    {'interval': 'hour',      'lookback_days': 300,  'output_dir': DATA_DIRS['1hour'],    'min_bars_required': 150},
    '5minute': {'interval': '5minute', 'lookback_days': 60, 'output_dir': os.path.join(BASE_DIR, "five_minute_data"), 'min_bars_required': 100}
}

RATE_LIMIT_DELAY = 0.5   # seconds between API calls
MAX_RETRIES = 3
RETRY_BACKOFF = 2        # exponential backoff multiplier
GOOGLE_SHEET_ID = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"

# -------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


class SmartHalalFetcher:
    def __init__(self):
        self.kite = None
        self.token_map = {}
        self.halal_symbols = []
        self.execution_stats = {
            'total_symbols': 0,
            'successful_fetches': 0,
            'failed_fetches': 0
        }
        self._setup_dirs()
        self._setup_kite()
        self._load_instruments()
        self._load_halal_symbols()

    def _setup_dirs(self):
        for d in DATA_DIRS.values():
            os.makedirs(d, exist_ok=True)

    def _setup_kite(self):
        """Load access_token and use API key from config."""
        try:
            # Import here to avoid circular imports
            from config import Config
            config = Config()
            
            # Try to load existing access token from file
            access_token = None
            if os.path.exists(TOKENS_FILE):
                try:
                    with open(TOKENS_FILE) as f:
                        token_data = json.load(f)
                    access_token = token_data.get("access_token")
                except Exception as e:
                    logging.warning(f"Could not read token file: {e}")
            
            # Initialize KiteConnect with API key from config
            self.kite = KiteConnect(api_key=config.API_KEY)
            
            if access_token:
                # Try using saved token first
                self.kite.set_access_token(access_token)
                try:
                    profile = self.kite.profile()
                    logging.info(f"✅ Using saved token. Authenticated as: {profile['user_name']}")
                    return
                except Exception:
                    logging.warning("Saved token invalid, need fresh authentication")
            
            # If no valid token, trigger authentication from config
            logging.info("No valid token found. Running authentication...")
            config.authenticate()  # This will prompt for login if needed
            self.kite = config.kite  # Use the authenticated kite instance
            profile = self.kite.profile()
            logging.info(f"✅ Authentication successful. Connected as: {profile['user_name']}")
            
        except Exception as e:
            raise Exception(f"Kite setup failed: {e}")

    def _load_instruments(self):
        instruments = self.kite.instruments("NSE")
        self.token_map = {inst["tradingsymbol"]: inst["instrument_token"]
                          for inst in instruments if inst['instrument_type'] == 'EQ'}
        logging.info(f"✅ Loaded {len(self.token_map)} NSE equity instruments")

    def _load_halal_symbols(self):
        halal_list = get_halal_list(GOOGLE_SHEET_ID)
        self.halal_symbols = [s for s in halal_list if s in self.token_map]
        self.execution_stats['total_symbols'] = len(self.halal_symbols)
        logging.info(f"✅ Halal equity symbols: {len(self.halal_symbols)}")

    def _clear_dir_with_backup(self, path):
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.csv')]
            if files:
                backup_dir = os.path.join(DATA_DIRS['backup'], f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(backup_dir, exist_ok=True)
                for file in files:
                    shutil.copy2(os.path.join(path, file), backup_dir)
                logging.info(f"📦 Backed up {len(files)} files to {backup_dir}")
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def _calculate_features(self, df, tf):
        """Calculate technical indicators with None handling."""
        try:
            params = {
                'daily':    {'rsi': 14, 'ema_fast': 10, 'ema_slow': 21, 'atr': 14},
                '15minute': {'rsi': 9,  'ema_fast': 5,  'ema_slow': 13, 'atr': 9},
                '1hour':    {'rsi': 14, 'ema_fast': 9,  'ema_slow': 21, 'atr': 14},
                '5minute':  {'rsi': 14, 'ema_fast': 5,  'ema_slow': 20, 'atr': 14}
            }[tf]

            # Calculate indicators with error handling
            df['rsi'] = ta.rsi(df['close'], length=params['rsi'])
            df['ema_fast'] = ta.ema(df['close'], length=params['ema_fast'])
            df['ema_slow'] = ta.ema(df['close'], length=params['ema_slow'])
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=params['atr'])
            
            # ADX calculation with error handling
            adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_result is not None and isinstance(adx_result, pd.DataFrame):
                df['adx'] = adx_result['ADX_14'] if 'ADX_14' in adx_result.columns else None
            else:
                df['adx'] = None

            # VWAP for intraday timeframes
            if tf in ['15minute', '1hour']:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                cumulative_volume = df['volume'].cumsum()
                # Avoid division by zero
                df['vwap'] = ((typical_price * df['volume']).cumsum() / 
                             cumulative_volume).ffill()

            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Fill NaN values with forward fill then backward fill
            df = df.ffill().bfill()
            
            return df
        except Exception as e:
            logging.error(f"Feature calculation error: {e}")
            return df

    def _fetch_symbol_timeframe(self, symbol, token, tf_name, cfg):
        to_dt = datetime.now()
        output_file = os.path.join(cfg['output_dir'], f"{symbol}.csv")
        
        if os.path.exists(output_file):
            try:
                df_existing = pd.read_csv(output_file)
                df_existing['date'] = pd.to_datetime(df_existing['date'])
                last_date = df_existing['date'].max()
                # Add one interval to last_date to avoid duplicate candles
                if tf_name == 'daily':
                    start_date = last_date + pd.Timedelta(days=1)
                elif tf_name == '15minute':
                    start_date = last_date + pd.Timedelta(minutes=15)
                elif tf_name == '1hour':
                    start_date = last_date + pd.Timedelta(hours=1)
                elif tf_name == '5minute':
                    start_date = last_date + pd.Timedelta(minutes=5)
                else:
                    start_date = last_date
            except Exception as e:
                logging.warning(f"{symbol}-{tf_name}: Failed to read existing data, fetching full range: {e}")
                start_date = to_dt - timedelta(days=cfg['lookback_days'])
        else:
            start_date = to_dt - timedelta(days=cfg['lookback_days'])
        
        if start_date >= to_dt:
            logging.info(f"{symbol}-{tf_name}: Data already up-to-date.")
            return True  # No need to fetch
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                time.sleep(RATE_LIMIT_DELAY)
                candles = self.kite.historical_data(token, start_date, to_dt, cfg['interval'])
                
                if not candles:
                    logging.info(f"{symbol}-{tf_name}: No new candles to fetch.")
                    return True
                
                df_new = pd.DataFrame(candles)
                df_new.rename(columns={'date': 'datetime'}, inplace=True)
                df_new['date'] = pd.to_datetime(df_new['datetime']).dt.tz_localize(None)
                
                df_new = self._calculate_features(df_new, tf_name)
                
                save_cols = [c for c in ['date','open','high','low','close','volume',
                                         'rsi','atr','adx','ema_fast','ema_slow',
                                         'volume_sma','volume_ratio','vwap'] if c in df_new.columns]
                df_out = df_new[save_cols].copy()
                
                if os.path.exists(output_file):
                    # Append only new rows to avoid duplicates
                    df_out = df_out[df_out['date'] > last_date]
                    if not df_out.empty:
                        df_existing = pd.read_csv(output_file)
                        df_updated = pd.concat([df_existing, df_out])
                        df_updated.to_csv(output_file, index=False)
                        logging.info(f"{symbol}-{tf_name}: Appended {len(df_out)} new candles.")
                    else:
                        logging.info(f"{symbol}-{tf_name}: No new candles after last date.")
                else:
                    df_out.to_csv(output_file, index=False)
                    logging.info(f"{symbol}-{tf_name}: Saved full data ({len(df_out)} candles).")
                
                return True
            except Exception as e:
                if "Too many requests" in str(e) or "429" in str(e):
                    sleep_time = RETRY_BACKOFF ** attempt
                    logging.warning(f"{symbol}-{tf_name}: Rate limited, sleeping {sleep_time}s")
                    time.sleep(sleep_time)
                else:
                    logging.warning(f"{symbol}-{tf_name} attempt {attempt}: {e}")
                    time.sleep(RETRY_BACKOFF ** attempt)
        
        logging.error(f"{symbol}-{tf_name}: Failed after {MAX_RETRIES} attempts")
        return False


    def fetch_all(self):
        """Fetch data for all symbols and timeframes."""
        logging.info("🧹 Clearing old data and creating backups...")
        for tf_name, cfg in SMART_TIMEFRAMES.items():
            self._clear_dir_with_backup(cfg['output_dir'])

        logging.info(f"📊 Starting data fetch for {len(self.halal_symbols)} symbols...")
        
        for i, symbol in enumerate(self.halal_symbols, 1):
            logging.info(f"[{i}/{len(self.halal_symbols)}] Processing {symbol}...")
            token = self.token_map.get(symbol)
            if not token:
                logging.warning(f"{symbol}: No token found")
                continue

            ok_count = 0
            for tf_name, cfg in SMART_TIMEFRAMES.items():
                if self._fetch_symbol_timeframe(symbol, token, tf_name, cfg):
                    ok_count += 1

            if ok_count > 0:
                self.execution_stats['successful_fetches'] += 1
            else:
                self.execution_stats['failed_fetches'] += 1

        self._print_summary()

    def _print_summary(self):
        elapsed = datetime.now() - datetime.now()  # You can track start time if needed
        logging.info("\n" + "="*50)
        logging.info("📈 DATA FETCH SUMMARY")
        logging.info("="*50)
        logging.info(f"Total symbols processed: {self.execution_stats['total_symbols']}")
        logging.info(f"Successfully fetched: {self.execution_stats['successful_fetches']}")
        logging.info(f"Failed to fetch: {self.execution_stats['failed_fetches']}")
        
        for tf_name, cfg in SMART_TIMEFRAMES.items():
            count = len([f for f in os.listdir(cfg['output_dir']) if f.endswith('.csv')])
            logging.info(f"📁 {tf_name}: {count} CSV files saved")
        
        logging.info("="*50)


def run_daily_refresh():
    """
    Refresh all configured timeframes for the Halal equity universe and
    write CSVs into DATA_DIRS while keeping a backup of previous files.
    This wraps the existing SmartHalalFetcher workflow for external callers.
    """
    logging.info("Starting daily refresh via run_daily_refresh()")
    fetcher = SmartHalalFetcher()
    fetcher.fetch_all()
    logging.info("Daily refresh completed successfully")


def main():
    """
    Backward-compatible CLI entrypoint.
    """
    run_daily_refresh()


if __name__ == "__main__":
    main()
