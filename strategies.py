def sma_strategy(df, idx):
    sma20 = df["close"].rolling(20).mean().iloc[idx]
    close = df["close"].iloc[idx]
    return (close > sma20, close < sma20)
