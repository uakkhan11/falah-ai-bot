#!/usr/bin/env python3
# live_orchestrator.py
# Unified daily orchestrator for EMA(5/20) long-only strategy with EMA200/ADX gate,
# risk-managed portfolio, Zerodha execution, journaling, Sheets/Telegram, and dashboard hooks.

import os, glob, json, math, time, logging
from datetime import datetime
import pandas as pd
import numpy as np

# ---------- Imports from your repo ----------
from config import Config                                     # broker + creds [file:367]
from improved_fetcher import BASE_DIR, DATA_DIRS, run_daily_refresh as fetch_data     # data refresh [file:358]
from order_manager import OrderManager                        # place/cancel orders [file:365]
from order_tracker import OrderTracker                        # track order status [file:373]
from holding_tracker import HoldingTracker                    # local positions state [file:370]
from capital_manager import CapitalManager                    # position sizing utils [file:368]
from risk_manager import RiskManager                          # kill-switch, gates [file:375]
from trade_logger import TradeLogger                          # CSV journaling [file:374]
from exit_manager import ExitManager                          # broker-hosted SL/GTT [file:380]
from gsheet_manager import GoogleSheetLogger                  # mobile journal [file:366]
from telegram_notifier import TelegramNotifier                # alerts [file:379]
# Optional intraday helpers (kept off by default)
# from live_data_manager import LiveDataManager               # LTP checks [file:372]
# from live_price_streamer import PriceStreamer               # streaming [file:378]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
STATE_DIR   = os.path.join(BASE_DIR, "state")
DAILY_DIR   = DATA_DIRS["daily"]
INDEX_PATH  = os.path.join(BASE_DIR, "index_data", "nifty_50.csv")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

DENY = {"NIFTY","NIFTY_50","NIFTY50","nifty_50","nifty50","NIFTY_INDEX"}

# ---------- Helpers ----------
def load_latest_params(params_json_override=""):
    if params_json_override and os.path.exists(params_json_override):
        with open(params_json_override) as f:
            return json.load(f), params_json_override
    files = sorted(glob.glob(os.path.join(REPORTS_DIR, "ema_daily_params_*.json")))
    if not files:
        raise SystemExit("No params JSON found in reports/. Run backtest_ema_daily.py to create one.")
    with open(files[-1]) as f:
        return json.load(f), files[-1]

def load_symbol_df(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for c in ["open","high","low","close"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in {path}")
    # Safety indicators if not present
    if "atr" not in df.columns or df["atr"].isna().all():
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"]  - df["close"].shift(1)).abs()
        tr  = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14, min_periods=14).mean()
    if "adx" not in df.columns or df["adx"].isna().all():
        up = df["high"].diff(); down = -df["low"].diff()
        plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"]  - df["close"].shift(1)).abs()
        tr  = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean().replace(0, np.nan)
        plus_di  = 100 * pd.Series(plus_dm).rolling(14, min_periods=14).sum() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(14, min_periods=14).sum() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        df["adx"] = dx.rolling(14, min_periods=14).mean()
    df["ema5"]   = df["close"].ewm(span=5, adjust=False).mean()
    df["ema20"]  = df["close"].ewm(span=20, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    return df

def load_index_gate(path):
    idx = pd.read_csv(path)
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    idx["ema200"] = idx["close"].ewm(span=200, adjust=False).mean()
    # same as backtest: 0.2% above EMA200
    idx["gate"] = idx["close"] > idx["ema200"] * 1.002
    return idx[["date","gate"]]

def build_today_intents(params, positions):
    # Unpack
    capital         = params["capital"]
    risk_per_trade  = params["risk_per_trade"]
    atr_mult        = params["atr_mult"]
    adx_thr         = params["adx_threshold"]
    trail           = params["trail"]
    ema_gap_pct     = params["ema_gap_pct"]
    ma_filter_pct   = params["ma_filter_pct"]
    min_atr_pct     = params["min_atr_pct"]
    min_hold_days   = params["min_hold_days"]
    max_new_per_day = params["max_new_per_day"]
    max_open        = params["max_open"]

    files = sorted(glob.glob(os.path.join(DAILY_DIR, "*.csv")))
    files = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in DENY]

    gate_df = load_index_gate(INDEX_PATH) if os.path.exists(INDEX_PATH) else None

    open_symbols = set(positions.keys())
    entries, exits = [], []

    for f in files:
        sym = os.path.basename(f).replace(".csv","")
        df = load_symbol_df(f)
        if len(df) < 202:
            continue

        # align index gate
        sym_gate = None
        if gate_df is not None:
            g = pd.merge(df[["date"]], gate_df, on="date", how="left")
            g["gate"] = g["gate"].ffill().fillna(False)
            sym_gate = g["gate"]

        # last closed bar
        i = len(df) - 1
        ema5_prev2 = df.loc[i-2,"ema5"];  ema20_prev2 = df.loc[i-2,"ema20"]
        ema5_prev  = df.loc[i-1,"ema5"];  ema20_prev  = df.loc[i-1,"ema20"]
        crossed_up   = (ema5_prev2 <= ema20_prev2) and (ema5_prev > ema20_prev)
        crossed_down = (ema5_prev2 >= ema20_prev2) and (ema5_prev < ema20_prev)
        c_prev  = df.loc[i-1,"close"]
        atr_prev= df.loc[i-1,"atr"]
        adx_prev= df.loc[i-1,"adx"] if not np.isnan(df.loc[i-1,"adx"]) else 0.0
        ema200_prev = df.loc[i-1,"ema200"]
        ref_date = str(df.loc[i-1,"date"].date())

        gap_ok = ((ema5_prev - ema20_prev) / max(1e-9, ema20_prev)) >= (ema_gap_pct/100.0)
        ma_ok  = ((c_prev - ema200_prev) / max(1e-9, ema200_prev)) >= (ma_filter_pct/100.0)
        atr_ok = (atr_prev / max(1e-9, c_prev)) >= (min_atr_pct/100.0) if not np.isnan(atr_prev) else False
        idx_ok = True if sym_gate is None else bool(sym_gate.iloc[-1])

        # Entry intent
        if sym not in open_symbols and crossed_up and gap_ok and ma_ok and atr_ok and adx_prev >= adx_thr and idx_ok:
            risk_ps = max(0.01, c_prev - (c_prev - atr_mult * atr_prev))
            qty = int(max(1, math.floor((risk_per_trade * capital) / risk_ps)))
            if qty > 0:
                entries.append({"symbol": sym, "qty": qty, "ref_date": ref_date})

        # Exit intent (cross-down exit; broker-hosted SL handles stop breaches intraday)
        if sym in open_symbols and crossed_down:
            exits.append({"symbol": sym, "reason": "XDOWN", "ref_date": ref_date})

    # enforce caps
    slots_free = max_open - len(open_symbols)
    if slots_free <= 0:
        entries = []
    else:
        entries = entries[: min(slots_free, max_new_per_day)]
    return entries, exits

# ---------- Main Orchestration ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", type=str, default="true", help="true/false")
    ap.add_argument("--params_json", type=str, default="", help="optional explicit params JSON path")
    ap.add_argument("--refresh_data", type=str, default="true", help="true/false refresh daily data")
    ap.add_argument("--notify", type=str, default="true", help="true/false Telegram")
    ap.add_argument("--push_sheet", type=str, default="true", help="true/false Google Sheets")
    args = ap.parse_args()

    dry_run  = args.dry_run.lower() == "true"
    do_fetch = args.refresh_data.lower() == "true"
    do_notify= args.notify.lower() == "true"
    do_sheet = args.push_sheet.lower() == "true"

    # Services
    cfg = Config()
    if not dry_run:
        cfg.authenticate()
    kite = getattr(cfg, "kite", None)
    
    # Core collaborators
    om = OrderManager(kite, cfg)                    # required by CapitalManager
    ot = OrderTracker(kite, cfg)                    # required by RiskManager and CapitalManager
    tl = TradeLogger(os.path.join(REPORTS_DIR, "live_trades.csv"))
    
    # Notifier (token + chat id pulled from Config)
    bot_token = getattr(cfg, "telegram_bot_token", None)
    chat_id   = getattr(cfg, "telegram_chat_id", None)
    tg = TelegramNotifier(bot_token, chat_id) if (do_notify and bot_token and chat_id) else None
    
    # Exit manager (needs om, tl, tg)
    em = ExitManager(kite, cfg, om, tl, tg)
    
    # State managers
    ht = HoldingTracker(os.path.join(STATE_DIR, "positions.json"))
    rm = RiskManager(os.path.join(STATE_DIR, "risk_state.json"), ot)
    
    # Capital manager now with required collaborators
    cm = CapitalManager(cfg, ot, om, tg)
    
    # Optional mobile journal
    gs = GoogleSheetLogger(cfg) if do_sheet else None

    # Data refresh
    if do_fetch:
        try:
            logging.info("Refreshing daily data...")
            fetch_data()
        except Exception as ex:
            logging.error(f"Data refresh failed: {ex}")
            if tg: tg.send_text(f"Data refresh failed: {ex}")

    # Load strategy parameters (from backtester)
    params, params_path = load_latest_params(args.params_json)
    logging.info(f"Using params: {os.path.basename(params_path)}")

    # Pre-trade risk gate (index trend, stale data, kill-switch)
    if not rm.pre_trade_ok(INDEX_PATH):
        msg = "RiskManager: blocked new entries (gate/kill-switch). Exits only."
        logging.warning(msg)
        if tg: tg.send_text(msg)

    # Sync positions (optional: broker reconciliation)
    positions = ht.load()
    # positions = ht.reconcile_with_broker(cfg.kite) if hasattr(ht, "reconcile_with_broker") else positions

    # Build today’s intents from last closed bar
    entries, exits = build_today_intents(params, positions)

    # Apply kill-switch to entries only
    if rm.is_paused():
        entries = []

    # Capital sizing hook (if capping by rupee or liquidity)
    for e in entries:
        e["qty"] = cm.size_quantity(e["symbol"], e["qty"])

    placed_entries, placed_exits = [], []

    # Place entries
    for e in entries:
        # If CapitalManager has size_quantity
            if hasattr(cm, "size_quantity"):
                e["qty"] = cm.size_quantity(e["symbol"], e["qty"])
            # Else if it has size_order using c_prev and stop distance:
            else:
                # pull last close and atr to compute stop distance akin to backtest sizing
                df_sym = load_symbol_df(os.path.join(DAILY_DIR, f"{e['symbol']}.csv"))
                c_prev = float(df_sym["close"].iloc[-1])
                atr_prev = float(df_sym["atr"].iloc[-1])
                stop_dist = params["atr_mult"] * atr_prev
                sized = cm.size_order(e["symbol"], entry_price=c_prev, stop_distance=stop_dist)
                e["qty"] = int(sized.get("qty", e["qty"]))
            if filled:
                ht.add(sym, qty, entry_price=avg_price)
                # place broker-hosted SL (GTT) at ATR-multiple below entry
                # Use last ATR from DF for safety
                df = load_symbol_df(os.path.join(DAILY_DIR, f"{sym}.csv"))
                atr_prev = float(df["atr"].iloc[-1])
                entry_p  = avg_price if avg_price else float(df["close"].iloc[-1])
                trigger  = round(entry_p - params["atr_mult"] * atr_prev, 2)
                if not dry_run and trigger > 0:
                    gtt_id = em.create_stop(sym, qty, trigger_price=trigger)
                    ht.update_gtt(sym, gtt_id)
            else:
                logging.warning(f"Entry not filled for {sym} (order_id={order_id})")
        except Exception as ex:
            logging.error(f"BUY failed {sym}: {ex}")
            if tg: tg.send_text(f"BUY failed {sym}: {ex}")

    # Place exits
    for x in exits:
        sym = x["symbol"]
        qty = ht.get_qty(sym)
        if qty <= 0:
            continue
        try:
            # cancel GTT before manual exit
            if not dry_run:
                gtt_id = ht.get_gtt(sym)
                if gtt_id:
                    em.cancel_stop(gtt_id)
            if dry_run:
                order_id = "DRYRUN"
            else:
                order_id = om.sell_market(sym, qty)
                # optionally wait for completion
                _ = ot.wait_for_complete(order_id, timeout_sec=30)
            placed_exits.append({**x, "order_id": order_id, "qty": qty})
            ht.remove(sym)
        except Exception as ex:
            logging.error(f"SELL failed {sym}: {ex}")
            if tg: tg.send_text(f"SELL failed {sym}: {ex}")

    # Persist positions
    ht.save()

    # Journal trades
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for rec in placed_entries:
        tl.log(now, rec["symbol"], "BUY", rec.get("qty",0), rec.get("order_id",""), note="daily-entry")
    for rec in placed_exits:
        tl.log(now, rec["symbol"], "SELL", rec.get("qty",0), rec.get("order_id",""), note=rec.get("reason","exit"))

    # Sheets summary row
    if gs:
        try:
            gs.append_row({
                "timestamp": now,
                "mode": "DRY" if dry_run else "LIVE",
                "entries": len(placed_entries),
                "exits": len(placed_exits),
                "open_positions": len(ht.load()),
                "params": os.path.basename(params_path)
            })
        except Exception as ex:
            logging.error(f"Sheet push failed: {ex}")

    # Telegram summary
    if tg:
        tg.send_text(f"Run {'DRY' if dry_run else 'LIVE'} | Entries {len(placed_entries)} | Exits {len(placed_exits)} | Open {len(ht.load())} | {os.path.basename(params_path)}")

    logging.info("Live orchestration completed.")

if __name__ == "__main__":
    main()
