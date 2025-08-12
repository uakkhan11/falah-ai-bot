#!/usr/bin/env python3
import os
import time
import json
import shutil
import logging
from datetime import datetime, timedelta, time as dt_time

import pandas as pd
import pandas_ta as ta
from kiteconnect import KiteConnect
from utils import get_halal_list

# -------------------- CONFIG --------------------
BASE_DIR = "/root/falah-ai-bot"
TOKENS_FILE = os.path.join(BASE_DIR, "kite_tokens.json")  # Token file from fetch_token.py

DATA_DIRS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    'backup': os.path.join(BASE_DIR, "data_backup")
}

SMART_TIMEFRAMES = {
    'daily':    {'interval': 'day',       'lookback_days': 1825, 'output_dir': DATA_DIRS['daily'],     'min_bars_required': 200},
    '15minute': {'interval': '15minute',  'lookback_days': 120,  'output_dir': DATA_DIRS['15minute'], 'min_bars_required': 100},
    '1hour':    {'interval': 'hour',      'lookback_days': 300,  'output_dir': DATA_DIRS['1hour'],    'min_bars_required': 150}
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
        """Load API key + access_token from saved file and init KiteConnect."""
        if not os.path.exists(TOKENS_FILE):
            raise FileNotFoundError(f"Token file {TOKENS_FILE} not found. Please run fetch_token.py first.")
        with open(TOKENS_FILE) as f:
            token_data = json.load(f)

        api_key = token_data.get("api_key") or token_data.get("API_KEY")
        access_token = token_data.get("access_token")
        if not api_key or not access_token:
            raise ValueError("Token file missing api_key or access_token.")

        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        profile = self.kite.profile()
        logging.info(f"✅ Authenticated to Kite as: {profile['user_name']}")

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
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def _calculate_features(self, df, tf):
        params = {
            'daily':    {'rsi': 14, 'ema_fast': 10, 'ema_slow': 21, 'atr': 14},
            '15minute': {'rsi': 9,  'ema_fast': 5,  'ema_slow': 13, 'atr': 9},
            '1hour':    {'rsi': 14, 'ema_fast': 9,  'ema_slow': 21, 'atr': 14}
        }[tf]
        df['rsi'] = ta.rsi(df['close'], length=params['rsi'])
        df['ema_fast'] = ta.ema(df['close'], length=params['ema_fast'])
        df['ema_slow'] = ta.ema(df['close'], length=params['ema_slow'])
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=params['atr'])
        df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        if tf in ['15minute', '1hour']:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        return df

    def _fetch_symbol_timeframe(self, symbol, token, tf_name, cfg):
        """Fetch single symbol/timeframe with retries."""
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=cfg['lookback_days'])

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                time.sleep(RATE_LIMIT_DELAY)
                candles = self.kite.historical_data(token, from_dt, to_dt, cfg['interval'])
                if not candles: return False
                df = pd.DataFrame(candles)
                if len(df) < cfg['min_bars_required']: return False
                df.rename(columns={'date': 'datetime'}, inplace=True)
                df['date'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
                df = self._calculate_features(df, tf_name)
                save_cols = [c for c in ['date','open','high','low','close','volume',
                                         'rsi','atr','adx','ema_fast','ema_slow',
                                         'volume_sma','volume_ratio','vwap'] if c in df.columns]
                df[save_cols].to_csv(os.path.join(cfg['output_dir'], f"{symbol}.csv"), index=False)
                return True
            except Exception as e:
                logging.warning(f"{symbol}-{tf_name} attempt {attempt} failed: {e}")
                time.sleep(RETRY_BACKOFF ** attempt)
        return False

    def fetch_all(self):
        for tf_name, cfg in SMART_TIMEFRAMES.items():
            self._clear_dir_with_backup(cfg['output_dir'])

        for symbol in self.halal_symbols:
            token = self.token_map.get(symbol)
            ok_count = 0
            for tf_name, cfg in SMART_TIMEFRAMES.items():
                if self._fetch_symbol_timeframe(symbol, token, tf_name, cfg):
                    ok_count += 1
            if ok_count:
                self.execution_stats['successful_fetches'] += 1
            else:
                self.execution_stats['failed_fetches'] += 1
        self._print_summary()

    def _print_summary(self):
        logging.info("\n--- Summary ---")
        logging.info(f"Total symbols: {self.execution_stats['total_symbols']}")
        logging.info(f"Successful: {self.execution_stats['successful_fetches']}")
        logging.info(f"Failed: {self.execution_stats['failed_fetches']}")
        for tf_name, cfg in SMART_TIMEFRAMES.items():
            count = len([f for f in os.listdir(cfg['output_dir']) if f.endswith('.csv')])
            logging.info(f"{tf_name}: {count} CSVs")


if __name__ == "__main__":
    fetcher = SmartHalalFetcher()
    fetcher.fetch_all()
