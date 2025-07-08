from datetime import datetime

def calculate_ai_exit_score(stock_data, trailing_sl, current_price, atr_value=None):
    score = 0
    reasons = []
    last = stock_data.iloc[-1]
    recent_vol = last["Volume"]
    avg_vol = stock_data["Volume"].tail(10).mean()

    if current_price < trailing_sl:
        drop_pct = (trailing_sl - current_price) / trailing_sl
        penalty = int(drop_pct * 100)
        score -= penalty
        reasons.append(f"Trailing SL hit (-{penalty} pts)")

    if "Supertrend" in stock_data.columns:
        if current_price < stock_data["Supertrend"].iloc[-1]:
            score -= 25
            reasons.append("Supertrend reversal")

    if recent_vol < 0.7 * avg_vol:
        score -= 10
        reasons.append("Volume drop")

    if (last["Close"] < last["Open"]) and ((last["Open"] - last["Close"]) > 0.005 * last["Open"]):
        score -= 15
        reasons.append("Bearish candle")

    if (recent_vol > 1.5 * avg_vol) and (last["Close"] < last["Open"]):
        score -= 15
        reasons.append("High-volume bearish candle")

    now = datetime.now()
    if now.hour == 15 and now.minute >= 0:
        score -= 20
        reasons.append("End of day exit window")

    if (last["Close"] > last["Open"]) and ((last["Close"] - last["Open"]) > 0.01 * last["Open"]):
        score += 10
        reasons.append("Strong bullish close")

    if "ADX_14" in stock_data.columns:
        if stock_data["ADX_14"].iloc[-1] < 20:
            score -= 10
            reasons.append("Weak trend (low ADX)")

    if atr_value:
        atr_distance = abs(current_price - trailing_sl) / atr_value
        if atr_distance < 0.5:
            score -= 10
            reasons.append("Price near trailing SL")

    gap_pct = (last["Open"] - stock_data["Close"].iloc[-2]) / stock_data["Close"].iloc[-2]
    if gap_pct < -0.01:
        score -= 10
        reasons.append("Gap down open")

    return score, reasons
