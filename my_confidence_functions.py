def compute_confidence_score(stock_data):
    """
    Compute numeric confidence score (0.0 - 1.0) based on multiple factors.
    """
    score = 0

    # Trend strength
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

    return min(score, 1.0)


def compute_allocation_weight(confidence_score):
    """
    Return a numeric weight multiplier (e.g., 0.8 to 1.5) based on confidence.
    """
    if confidence_score >= 0.9:
        return 1.5
    elif confidence_score >= 0.75:
        return 1.3
    elif confidence_score >= 0.6:
        return 1.1
    else:
        return 0.8


def adjust_capital_based_on_confidence(total_capital, confidence_score):
    """
    Allocate more capital to higher confidence setups.
    For example:
        - confidence 0.9+ => 1.5x base
        - confidence 0.7-0.9 => 1.2x base
        - confidence <0.7 => base
    """
    base = total_capital / 10  # Example: split into 10 parts
    if confidence_score >= 0.9:
        return base * 1.5
    elif confidence_score >= 0.7:
        return base * 1.2
    else:
        return base
