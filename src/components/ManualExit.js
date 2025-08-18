import React, { useState } from "react";

export default function ManualExit() {
  const [symbol, setSymbol] = useState("");
  const [message, setMessage] = useState("");

  const exitPosition = () => {
    if (!symbol) {
      setMessage("Please enter a symbol");
      return;
    }

    fetch("http://localhost:8000/api/trade/exit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: symbol.toUpperCase() }),
    })
      .then(res => res.json())
      .then(data => setMessage(JSON.stringify(data)))
      .catch(error => setMessage("Error: " + error.message));
  };

  return (
    <div>
      <h2>Manual Exit (Close Position)</h2>
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
      <button onClick={exitPosition}>Exit Position</button>
      <div>{message}</div>
    </div>
  );
}
