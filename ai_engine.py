from datetime import datetime
import pytz
from ta.trend import ADXIndicator

def calculate_ai_exit_score(stock_data, trailing_sl, current_price, atr_value=None):
    score = 0
    reasons = []

    # ✅ Ensure ADX column exists
    if "ADX" not in stock_data.columns:
        adx_indicator = ADXIndicator(
            high=stock_data["high"],
            low=stock_data["low"],
            close=stock_data["close"],
            window=14
        )
        stock_data["ADX"] = adx_indicator.adx()

    last = stock_data.iloc[-1]

    # ✅ Handle volume column flexibility
    volume_col = None
    for col in ["Volume", "volume"]:
        if col in stock_data.columns:
            volume_col = col
            break

    recent_vol = last[volume_col] if volume_col else None
    avg_vol = stock_data[volume_col].tail(10).mean() if volume_col else None

    # ✅ 1. Trailing Stop Loss
    if current_price < trailing_sl:
        drop_pct = (trailing_sl - current_price) / trailing_sl
        penalty = int(drop_pct * 100)
        score -= penalty
        reasons.append(f"Trailing SL hit (-{penalty} pts)")

    # ✅ 2. Supertrend
    if "Supertrend" in last and current_price < last["Supertrend"]:
        score -= 25
        reasons.append("Supertrend reversal")

    # ✅ 3. Volume drop
    if recent_vol is not None and recent_vol < 0.7 * avg_vol:
        score -= 10
        reasons.append("Volume drop")

    # ✅ 4. Bearish candle
    if last["close"] < last["open"] and (last["open"] - last["close"]) > 0.005 * last["open"]:
        score -= 15
        reasons.append("Bearish candle")

    # ✅ 5. High-volume bearish candle
    if recent_vol is not None and recent_vol > 1.5 * avg_vol and last["close"] < last["open"]:
        score -= 15
        reasons.append("High-volume bearish candle")

    # ✅ 6. End of day penalty
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    if now.hour == 15:
        score -= 20
        reasons.append("End of day exit window")

    # ✅ 7. Strong bullish close
    if last["close"] > last["open"] and (last["close"] - last["open"]) > 0.01 * last["open"]:
        score += 10
        reasons.append("Strong bullish close")

    # ✅ 8. ADX weak trend detection
    adx_col = None
    for col in ["ADX_14", "ADX"]:
        if col in stock_data.columns:
            adx_col = col
            break
    if adx_col and stock_data[adx_col].iloc[-1] < 20:
        score -= 10
        reasons.append("Weak trend (low ADX)")

    # ✅ 9. ATR proximity to trailing SL
    if atr_value:
        atr_distance = abs(current_price - trailing_sl) / atr_value
        if atr_distance < 0.5:
            score -= 10
            reasons.append("Price near trailing SL")

    # ✅ 10. Gap down open
    if len(stock_data) >= 2:
        prev_close = stock_data.iloc[-2]["close"]
        gap_pct = (last["open"] - prev_close) / prev_close
        if gap_pct < -0.01:
            score -= 10
            reasons.append("Gap down open")

    return score, reasons
