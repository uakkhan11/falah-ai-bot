import React, { useState } from "react";

export default function ManualTrade() {
  const [symbol, setSymbol] = useState("");
  const [qty, setQty] = useState("");
  const [tradeType, setTradeType] = useState("buy");
  const [message, setMessage] = useState("");

  const placeOrder = () => {
    if (!symbol || !qty || qty <= 0) {
      setMessage("Please enter valid symbol and quantity");
      return;
    }
    fetch(`http://localhost:8000/api/trade/${tradeType}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: symbol.toUpperCase(), qty: Number(qty) }),
    })
      .then(res => res.json())
      .then(data => setMessage(JSON.stringify(data)))
      .catch(error => setMessage("Error: " + error.message));
  };

  return (
    <div>
      <h2>Manual Trade</h2>
      <div>
        <label>
          Symbol:{" "}
          <input
            value={symbol}
            onChange={e => setSymbol(e.target.value.toUpperCase())}
            placeholder="e.g., INFIBEAM"
          />
        </label>
      </div>
      <div>
        <label>
          Quantity:{" "}
          <input
            type="number"
            value={qty}
            onChange={e => setQty(e.target.value)}
            placeholder="e.g., 100"
          />
        </label>
      </div>
      <div>
        <label>
          Trade Type:{" "}
          <select value={tradeType} onChange={e => setTradeType(e.target.value)}>
            <option value="buy">Buy</option>
            <option value="sell">Sell</option>
          </select>
        </label>
      </div>
      <button onClick={placeOrder}>Place Order</button>
      <div>{message}</div>
    </div>
  );
}
