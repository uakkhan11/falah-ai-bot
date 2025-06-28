# smart_scanner.py – Multi-Timeframe, Multi-Threaded Halal Scanner
from utils import get_halal_list

import threading
import pandas as pd
import pytz
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from utils import load_credentials, get_halal_list
from indicators import detect_breakout, detect_rsi_ema_signals, detect_3green_days, detect_darvas_box

# ---- Init ----
IST = pytz.timezone('Asia/Kolkata')
secrets = load_credentials()
kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])
kite.set_access_token(secrets["zerodha"]["access_token"])

# ---- Config ----
HALAL_LIST_PATH = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"
results_lock = threading.Lock()
scan_results = []
