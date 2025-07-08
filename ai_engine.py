# ai_engine.py

from datetime import datetime

def calculate_ai_exit_score(stock_data, trailing_sl, current_price, atr_value=None):
    """
    Computes an AI-based exit score for a given stock.

    Args:
        stock_data (DataFrame): Historical OHLCV data with indicators.
        trailing_sl (float): The current trailing stoploss level.
        current_price (float): Latest market price.
        atr_value (float): Latest ATR value for distance calculations.

    Returns:
        Tuple[int, List[str]]: (Score, List of reasons contributing to the score)
    """
    score = 0
    reasons = []
    last = stock_data.iloc[-1]
    recent_vol = last["Volume"]
    avg_vol = stock_data["Volume"].tail(10).mean()

    # 🔻 1. Trailing SL breach (dynamic penalty)
    if current_price < trailing_sl:
        drop_pct = (trailing_sl - current_price) / trailing_sl
        penalty = int(drop_pct * 100)
        score -= penalty
        reasons.append(f"Trailing SL hit (-{penalty} pts)")

    # 📉 2. Supertrend reversal
    if "Supertrend" in stock_data.columns:
        if current_price < stock_data["Supertrend"].iloc[-1]:
            score -= 25
            reasons.append("Supertrend reversal")

    # 📊 3. Volume drop
    if recent_vol < 0.7 * avg_vol:
        score -= 10
        reasons.append("Volume drop")

    # 🧠 4. Bearish reversal candle
    if (last["Close"] < last["Open"]) and ((last["Open"] - last["Close"]) > 0.005 * last["Open"]):
        score -= 15
        reasons.append("Bearish candle")

    # ⚡ 5. High-volume bearish candle
    if (recent_vol > 1.5 * avg_vol) and (last["Close"] < last["Open"]):
        score -= 15
        reasons.append("High-volume bearish candle")

    # 🕒 6. End of day exit window
    now = datetime.now()
    if now.hour == 15 and now.minute >= 0:
        score -= 20
        reasons.append("End of day exit window")

    # ✅ 7. Bonus: Strong bullish close
    if (last["Close"] > last["Open"]) and ((last["Close"] - last["Open"]) > 0.01 * last["Open"]):
        score += 10
        reasons.append("Strong bullish close")

    # 📈 8. Weak trend detected (ADX)
    if "ADX_14" in stock_data.columns:
        if stock_data["ADX_14"].iloc[-1] < 20:
            score -= 10
            reasons.append("Weak trend (low ADX)")

    # 🔄 9. ATR distance to trailing SL
    if atr_value:
        atr_distance = abs(current_price - trailing_sl) / atr_value
        if atr_distance < 0.5:
            score -= 10
            reasons.append("Price near trailing SL")

    # 📉 10. Gap down open
    gap_pct = (last["Open"] - stock_data["Close"].iloc[-2]) / stock_data["Close"].iloc[-2]
    if gap_pct < -0.01:
        score -= 10
        reasons.append("Gap down open")

    return score, reasons


def analyze_exit_signals(symbol, avg_price, cmp):
    """
    Basic AI exit signal logic.
    Returns True if CMP has dropped more than 3% below average price.
    """
    if cmp < avg_price * 0.97:
        return True
    return False
