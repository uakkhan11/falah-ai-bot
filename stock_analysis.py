import pandas as pd
import pandas_ta as ta
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator
from ai_engine import calculate_ai_exit_score

def get_regime(adx_value):
    return "TREND" if adx_value >= 25 else "RANGE"

def load_nifty_df():
    df_nifty = pd.read_csv("historical_data/NIFTY.csv")
    df_nifty["date"] = pd.to_datetime(df_nifty["date"])
    return df_nifty

def analyze_stock(kite, symbol):
    ltp_data = kite.ltp(f"NSE:{symbol}")
    cmp = ltp_data[f"NSE:{symbol}"]["last_price"]
    instrument_token = ltp_data[f"NSE:{symbol}"]["instrument_token"]

    from data_fetch import fetch_historical_candles
    hist = fetch_historical_candles(kite, instrument_token, interval="day", days=30)
    df = pd.DataFrame(hist)
    df.columns = [col.capitalize() for col in df.columns]

    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    adx_indicator = ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["ADX"] = adx_indicator.adx()
    rsi_series = ta.rsi(df["Close"], length=14)
    df["RSI"] = rsi_series
    rsi_latest = rsi_series.iloc[-1]
    rsi_percentile = (rsi_latest - rsi_series.min()) / (rsi_series.max() - rsi_series.min())
    boll = ta.bbands(df["Close"], length=20, std=2)
    df["BB_upper"] = boll["BBU_20_2.0"]
    df["BB_lower"] = boll["BBL_20_2.0"]
    df["BB_mid"] = boll["BBM_20_2.0"]
    supertrend_df = ta.supertrend(df["High"], df["Low"], df["Close"], length=10, multiplier=3.0)
    df["Supertrend"] = supertrend_df["SUPERT_10_3.0"]

    nifty_df = load_nifty_df()
    merged_df = df.merge(nifty_df, left_on="Date", right_on="date", suffixes=("", "_nifty"))
    merged_df["RelStrength"] = merged_df["Close"] / merged_df["Close_nifty"]
    latest_rel_strength = merged_df["RelStrength"].iloc[-1]

    trailing_sl = cmp - df["ATR"].iloc[-1] * 1.5
    ai_score, reasons = calculate_ai_exit_score(df, trailing_sl, cmp, atr_value=df["ATR"].iloc[-1])
    risk_amount = df["ATR"].iloc[-1] * 1.5
    reward_amount = df["ATR"].iloc[-1] * 3.0
    win_rate = (df.tail(10)["Close"] > df.tail(10)["Supertrend"]).mean() * 100

    recommendation = "Hold"
    if df["ADX"].iloc[-1] > 25 and rsi_latest < 70 and cmp > df["Supertrend"].iloc[-1] and latest_rel_strength > 1:
        recommendation = "Potential Buy"

    return {
        "cmp": cmp,
        "atr": df["ATR"].iloc[-1],
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
