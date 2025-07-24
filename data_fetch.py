def get_intraday_data(kite, symbol, interval="15minute", days=1):
    """
    Fetch intraday data for the given symbol using Zerodha Kite.
    - interval: '5minute', '15minute', etc.
    - days: number of past days (default 1 for intraday)
    """
    import pandas as pd
    from datetime import datetime, timedelta
    import pytz

    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    from_date = now - timedelta(days=days)
    to_date = now

    # Get instrument token
    instrument_token = None
    try:
        instrument_dump = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instrument_dump)
        row = instrument_df[instrument_df["tradingsymbol"] == symbol]
        if not row.empty:
            instrument_token = int(row.iloc[0]["instrument_token"])
    except Exception as e:
        print(f"⚠️ Failed to fetch instrument token for {symbol}: {e}")
        return None

    if not instrument_token:
        print(f"❌ Instrument token not found for {symbol}")
        return None

    try:
        data = kite.historical_data(
            instrument_token,
            from_date,
            to_date,
            interval=interval,
            continuous=False
        )
        df = pd.DataFrame(data)
        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"]).dt.tz_convert("Asia/Kolkata")
        df.set_index("date", inplace=True)
        return df

    except Exception as e:
        print(f"❌ Failed to fetch intraday data for {symbol}: {e}")
        return None

def get_live_ltp(kite, symbols):
    """
    Fetch live LTPs for a list of NSE symbols.
    Example: ['INFY', 'RELIANCE']
    Returns: {'INFY': 1525.0, 'RELIANCE': 2850.35}
    """
    try:
        full_symbols = [f"NSE:{symbol}" for symbol in symbols]
        ltp_data = kite.ltp(full_symbols)
        return {symbol.split(":")[1]: ltp_data[symbol]["last_price"] for symbol in ltp_data}
    except Exception as e:
        print(f"⚠️ Error fetching LTPs: {e}")
        return {}
