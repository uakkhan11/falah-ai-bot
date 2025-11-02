symbol = 'BHARTIARTL'  # Replace with your symbol
df = pd.read_csv(f'swingdata/{symbol}.csv')
df = supertrend(df, period=10, multiplier=3)

# Add SuperTrend values to the DataFrame
df = df[['date', 'close', 'upperband', 'lowerband', 'supertrend']]
df.to_csv(f'{symbol}_supertrend_export.csv', index=False)
