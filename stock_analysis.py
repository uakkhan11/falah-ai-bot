import pandas as pd
import pandas_ta as ta
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator
from ai_engine import calculate_ai_exit_score


def get_regime(adx_value):
    return "TREND" if adx_value >= 25 else "RANGE"


def load_nifty_df():
    df_nifty = pd.read_csv("historical_data/NIFTY.csv")
    df_nifty["date"] = pd.to_datetime(df_nifty["date"]).dt.tz_localize(None)
    return df_nifty


def analyze_stock(kite, symbol):
    # Get live LTP and instrument token
    ltp_data = kite.ltp(f"NSE:{symbol}")
    cmp = ltp_data[f"NSE:{symbol}"]["last_price"]
    instrument_token = ltp_data[f"NSE:{symbol}"]["instrument_token"]

    from data_fetch import fetch_historical_candles

    # Fetch historical data
    df = fetch_historical_candles(kite, instrument_token, interval="day", days=60)

    # Ensure correct column names
    df = df.rename(columns=str.lower)

    # Make sure date column exists and is datetime
    if "date" not in df:
        raise ValueError("Missing 'date' column in historical data.")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # Sort
    df = df.sort_values("date").reset_index(drop=True)

    # Compute indicators
    df["ATR"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    adx_indicator = ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["ADX"] = adx_indicator.adx()
    df["RSI"] = ta.rsi(df["close"], length=14)
    boll = ta.bbands(df["close"], length=20, std=2)
    df["BB_upper"] = boll["BBU_20_2.0"]
    df["BB_lower"] = boll["BBL_20_2.0"]
    df["BB_mid"] = boll["BBM_20_2.0"]
    supertrend_df = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["Supertrend"] = supertrend_df["SUPERT_10_3.0"]

    # Drop rows with NaNs in any indicator
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("No data after computing indicators. Not enough candles.")

    # Compute latest RSI percentile safely
    rsi_latest = df["RSI"].iloc[-1]
    rsi_min = df["RSI"].min()
    rsi_max = df["RSI"].max()
    rsi_percentile = (
        (rsi_latest - rsi_min) / (rsi_max - rsi_min)
        if rsi_max != rsi_min else 0.5
    )

    # Load NIFTY for relative strength
    nifty_df = load_nifty_df()
    nifty_df_renamed = nifty_df.rename(columns={"date": "date_nifty", "close": "close_nifty"})
    merged_df = pd.merge_asof(
        df.sort_values("date"),
        nifty_df_renamed[["date_nifty", "close_nifty"]].sort_values("date_nifty"),
        left_on="date",
        right_on="date_nifty",
        direction="backward"
    )

    if merged_df["close_nifty"].isnull().all():
        raise ValueError("Could not align any NIFTY closes with the symbol candles.")

    merged_df["RelStrength"] = merged_df["close"] / merged_df["close_nifty"]
    latest_rel_strength = merged_df["RelStrength"].iloc[-1]

    # Compute trailing SL
    atr_latest = df["ATR"].iloc[-1]
    trailing_sl = cmp - atr_latest * 1.5

    # AI exit score
    ai_score, reasons = calculate_ai_exit_score(df, trailing_sl, cmp, atr_value=atr_latest)

    # Risk / reward
    risk_amount = atr_latest * 1.5
    reward_amount = atr_latest * 3.0

    # Simple backtest: win rate = % of closes > supertrend over last 10 bars
    win_rate = (df.tail(10)["close"] > df.tail(10)["Supertrend"]).mean() * 100

    # Recommendation
    recommendation = "Hold"
    if (
        df["ADX"].iloc[-1] > 25
        and rsi_latest < 70
        and cmp > df["Supertrend"].iloc[-1]
        and latest_rel_strength > 1
    ):
        recommendation = "Potential Buy"

    return {
        "cmp": cmp,
        "atr": atr_latest,
        "trailing_sl": trailing_sl,
        "adx": df["ADX"].iloc[-1],
        "rsi": rsi_latest,
        "rsi_percentile": rsi_percentile,
        "supertrend": df["Supertrend"].iloc[-1],
        "bb_upper": df["BB_upper"].iloc[-1],
        "bb_lower": df["BB_lower"].iloc[-1],
        "bb_mid": df["BB_mid"].iloc[-1],
        "rel_strength": latest_rel_strength,
        "risk": risk_amount,
        "reward": reward_amount,
        "backtest_winrate": win_rate,
        "ai_score": ai_score,
        "reasons": reasons,
        "recommendation": recommendation,
        "history": df
    }
