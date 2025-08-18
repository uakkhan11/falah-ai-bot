import React, { useEffect, useState } from "react";

export default function OpenTrades() {
  const [trades, setTrades] = useState([]);

  useEffect(() => {
    fetch("http://localhost:8000/api/positions/open")
      .then(res => res.json())
      .then(setTrades)
      .catch(console.error);
  }, []);

  return (
    <div>
      <h2>Open Trades</h2>
      <table border="1" cellPadding="8" cellSpacing="0">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Quantity</th>
            <th>P&L</th>
            <th>Entry Price</th>
            <th>Current Price</th>
          </tr>
        </thead>
        <tbody>
          {trades.length === 0 && (
            <tr>
              <td colSpan="5">No Open Trades</td>
            </tr>
          )}
          {trades.map(({ symbol, qty, pnl, entry_price, current_price }) => (
            <tr key={symbol}>
              <td>{symbol}</td>
              <td>{qty}</td>
              <td>{pnl}</td>
              <td>{entry_price}</td>
              <td>{current_price}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
