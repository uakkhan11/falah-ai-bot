from stock_analysis import analyze_stock

def analyze_multiple_stocks(kite, symbols):
    results = []
    for sym in symbols:
        try:
            res = analyze_stock(kite, sym)
            res["symbol"] = sym
            results.append(res)
        except Exception as e:
            results.append({
                "symbol": sym,
                "error": str(e)
            })
    return results
