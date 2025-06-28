from ta.volatility import AverageTrueRange
import pandas as pd

def calculate_position_size(kite, symbol, total_capital, risk_percent=0.02, atr_multiplier=1.5):
    """
    Calculates quantity to buy based on ATR volatility.
    """
    risk_capital = total_capital * risk_percent
    instrument_token = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"]
    hist = kite.historical_data(
        instrument_token=instrument_token,
        from_date=pd.Timestamp.today()-pd.Timedelta(days=30),
        to_date=pd.Timestamp.today(),
        interval="day"
    )
    df = pd.DataFrame(hist)
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range().iloc[-1]
    cmp = df.iloc[-1]["close"]
    risk_per_share = atr * atr_multiplier
    if risk_per_share == 0:
        risk_per_share = cmp * 0.02
    qty = int(risk_capital / risk_per_share)
    return qty, cmp
def calculate_atr_trailing_sl(kite, symbol, cmp, atr_multiplier=1.5):
    """
    Calculates ATR-based trailing stoploss.
    """
    instrument_token = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"]
    hist = kite.historical_data(
        instrument_token=instrument_token,
        from_date=pd.Timestamp.today() - pd.Timedelta(days=30),
        to_date=pd.Timestamp.today(),
        interval="day"
    )
    df = pd.DataFrame(hist)
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range().iloc[-1]
    sl_price = round(cmp - atr * atr_multiplier, 2)
    return sl_price
