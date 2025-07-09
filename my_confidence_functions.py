def compute_confidence_score(stock_data):
    """
    Compute a confidence score (0-1) based on various factors.
    This is a placeholder. You can customize the logic.
    """
    score = 0

    # Example: Trend strength
    if stock_data["adx"] > 25:
        score += 0.3
    else:
        score += 0.1

    # RSI in healthy range
    if 40 <= stock_data["rsi"] <= 60:
        score += 0.3
    else:
        score += 0.1

    # Relative Strength
    if stock_data["rel_strength"] > 1:
        score += 0.3
    else:
        score += 0.1

    # Backtest Win Rate
    if stock_data["backtest_winrate"] > 60:
        score += 0.3
    else:
        score += 0.1

    return min(score, 1)


def adjust_capital_based_on_confidence(total_capital, confidence_score):
    """
    Allocate more capital to higher confidence setups.
    For example:
        - confidence 0.9+ => 1.5x base
        - confidence 0.7-0.9 => 1.2x base
        - confidence <0.7 => base
    """
    base = total_capital / 10  # Example base capital per trade

    if confidence_score >= 0.9:
        return base * 1.5
    elif confidence_score >= 0.7:
        return base * 1.2
    else:
        return base
