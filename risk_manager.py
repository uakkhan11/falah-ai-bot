#!/usr/bin/env python3
# risk_manager.py
# Portfolio-level risk manager: kill-switch, stale-data guard, and index trend gate.

import os
import json
import logging
from datetime import datetime, timedelta

import pandas as pd

DEFAULTS = {
    "paused": False,
    "pause_reason": "",
    "last_update": None,
    "dd_threshold_pct": 10.0,        # kill-switch threshold
    "current_drawdown_pct": 0.0,
    "stale_days_allowed": 2          # max days since last index bar
}

class RiskManager:
    """
    Minimal RiskManager used by the live orchestrator.
    Constructor signature kept simple to avoid wiring issues:
      RiskManager(state_path, order_tracker)

    order_tracker is injected for future use (e.g., cancel all on emergency),
    but not required for basic checks and can be None in dry runs.
    """

    def __init__(self, state_path, order_tracker):
        self.state_path = state_path
        self.order_tracker = order_tracker
        self.state = dict(DEFAULTS)
        self._load_state()

    # ---------- Persistence ----------
    def _load_state(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                self.state.update(data or {})
        except Exception as ex:
            logging.error(f"RiskManager: failed to load state: {ex}")

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            self.state["last_update"] = datetime.now().isoformat()
            with open(self.state_path, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as ex:
            logging.error(f"RiskManager: failed to save state: {ex}")

    # ---------- Public API ----------
    def is_paused(self):
        return bool(self.state.get("paused", False))

    def set_pause(self, paused=True, reason=""):
        self.state["paused"] = bool(paused)
        self.state["pause_reason"] = str(reason or "")
        self._save_state()

    def record_drawdown(self, drawdown_pct):
        """
        Update rolling drawdown metric. If it breaches threshold,
        engage kill-switch (pause new entries).
        """
        try:
            dd = float(drawdown_pct)
        except Exception:
            dd = 0.0
        self.state["current_drawdown_pct"] = dd
        thr = float(self.state.get("dd_threshold_pct", DEFAULTS["dd_threshold_pct"]))
        if dd >= thr and not self.is_paused():
            self.set_pause(True, reason=f"DD {dd:.2f}% >= threshold {thr:.2f}%")
        else:
            self._save_state()

    def pre_trade_ok(self, index_csv_path):
        """
        Global pre-trade check used by the orchestrator.
        Returns False if:
          - kill-switch is active,
          - index data is stale,
          - index trend gate is OFF (close <= 1.002 * EMA200).
        """
        if self.is_paused():
            return False

        # Index CSV must be recent
        if not index_csv_path or not os.path.exists(index_csv_path):
            logging.warning("RiskManager: index CSV missing; blocking new entries.")
            return False

        try:
            df = pd.read_csv(index_csv_path)
            if "date" not in df.columns or "close" not in df.columns:
                logging.warning("RiskManager: index CSV invalid; blocking new entries.")
                return False
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # Freshness
            last_dt = df["date"].iloc[-1]
            stale_days_allowed = int(self.state.get("stale_days_allowed", DEFAULTS["stale_days_allowed"]))
            if datetime.now().date() - last_dt.date() > timedelta(days=stale_days_allowed):
                logging.warning(f"RiskManager: index CSV stale (last {last_dt.date()}); blocking entries.")
                return False

            # Trend gate: close > EMA200 * 1.002 (same as backtest)
            df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
            last_close = float(df["close"].iloc[-1])
            last_ema200 = float(df["ema200"].iloc[-1])
            gate_on = last_close > last_ema200 * 1.002
            if not gate_on:
                logging.info("RiskManager: index gate OFF; blocking entries.")
                return False

            return True
        except Exception as ex:
            logging.error(f"RiskManager: pre_trade_ok failed: {ex}")
            return False
