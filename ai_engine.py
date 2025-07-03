# ai_engine.py

def calculate_ai_exit_score(stock_data, trailing_sl, current_price):
    """
    Computes an AI-based exit score for a given stock.

    Args:
        stock_data (DataFrame): Historical OHLCV data with indicators.
        trailing_sl (float): The current trailing stoploss level.
        current_price (float): Latest market price.

    Returns:
        Tuple[int, List[str]]: (Score, List of reasons contributing to the score)
    """
    score = 0
    reasons = []

    # 🔻 1. Trailing SL hit
    if current_price < trailing_sl:
        score -= 40
        reasons.append("Trailing SL hit")

    # 📉 2. Supertrend Reversal
    if "Supertrend" in stock_data.columns:
        if current_price < stock_data["Supertrend"].iloc[-1]:
            score -= 25
            reasons.append("Supertrend reversal")

    # 📊 3. Volume Drop
    if "Volume" in stock_data.columns:
        recent_vol = stock_data["Volume"].iloc[-1]
        avg_vol = stock_data["Volume"].tail(10).mean()
        if recent_vol < 0.7 * avg_vol:
            score -= 10
            reasons.append("Volume drop")

    # 🧠 4. Reversal Candles
    last = stock_data.iloc[-1]
    if (last["Close"] < last["Open"]) and ((last["Open"] - last["Close"]) > 0.005 * last["Open"]):
        score -= 15
        reasons.append("Bearish candle")

    # 🕒 5. End of day exit
    from datetime import datetime
    now = datetime.now()
    if now.hour == 15 and now.minute >= 15:
        score -= 20
        reasons.append("Time-based exit")

    # ✅ Bonus: Bullish support
    if (last["Close"] > last["Open"]) and ((last["Close"] - last["Open"]) > 0.01 * last["Open"]):
        score += 10
        reasons.append("Strong bullish close")

    return score, reasons
