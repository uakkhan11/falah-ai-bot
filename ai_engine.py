from datetime import datetime
import pytz
from ta.trend import ADXIndicator

def calculate_ai_exit_score(stock_data, trailing_sl, current_price, atr_value=None):
    score = 0
    reasons = []

    
    # --- Step 1: Calculate ADX dynamically ---
    if "ADX" not in stock_data.columns:
        adx_indicator = ADXIndicator(
            high=stock_data["high"],
            low=stock_data["low"],
            close=stock_data["close"],
            window=14
        )
        stock_data["ADX"] = adx_indicator.adx()

    last = stock_data.iloc[-1]
    
    # Use lowercase volume if needed
    volume_col = "Volume" if "Volume" in stock_data.columns else "volume"
    recent_vol = last[volume_col]
    avg_vol = stock_data[volume_col].tail(10).mean()

    # Trailing SL
    if current_price < trailing_sl:
        drop_pct = (trailing_sl - current_price) / trailing_sl
        penalty = int(drop_pct * 100)
        score -= penalty
        reasons.append(f"Trailing SL hit (-{penalty} pts)")

    # Supertrend
    if "Supertrend" in stock_data.columns:
        if current_price < last["Supertrend"]:
            score -= 25
            reasons.append("Supertrend reversal")

    # Volume drop
    if recent_vol < 0.7 * avg_vol:
        score -= 10
        reasons.append("Volume drop")

    # Bearish candle
    if (last["close"] < last["open"]) and ((last["open"] - last["close"]) > 0.005 * last["open"]):
        score -= 15
        reasons.append("Bearish candle")

    # High-volume bearish candle
    if (recent_vol > 1.5 * avg_vol) and (last["close"] < last["open"]):
        score -= 15
        reasons.append("High-volume bearish candle")

    # End of day penalty (timezone aware)
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    if now.hour == 15 and now.minute >= 0:
        score -= 20
        reasons.append("End of day exit window")

    # Strong bullish close
    if (last["close"] > last["open"]) and ((last["close"] - last["open"]) > 0.01 * last["open"]):
        score += 10
        reasons.append("Strong bullish close")

    # ADX check (try both column names)
    adx_col = "ADX_14" if "ADX_14" in stock_data.columns else ("ADX" if "ADX" in stock_data.columns else None)
    if adx_col:
        if stock_data[adx_col].iloc[-1] < 20:
            score -= 10
            reasons.append("Weak trend (low ADX)")

    # ATR proximity
    if atr_value:
        atr_distance = abs(current_price - trailing_sl) / atr_value
        if atr_distance < 0.5:
            score -= 10
            reasons.append("Price near trailing SL")

    # Gap down open
    gap_pct = (last["open"] - stock_data["close"].iloc[-2]) / stock_data["close"].iloc[-2]
    if gap_pct < -0.01:
        score -= 10
        reasons.append("Gap down open")

    return score, reasons
