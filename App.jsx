function App() {
  const trades = [
    { id: 1, symbol: "AAPL", quantity: 10, price: 192.38, status: "Open" },
    { id: 2, symbol: "TSLA", quantity: 5, price: 247.11, status: "Closed" },
    { id: 3, symbol: "BTCUSD", quantity: 0.15, price: 31000.5, status: "Open" },
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <h1 className="text-4xl font-bold text-blue-600 mb-6">Trades Table</h1>

      <div className="overflow-x-auto bg-white rounded shadow p-4">
        <table className="min-w-full table-auto">
          <thead>
            <tr className="bg-gray-200 text-gray-600 uppercase text-sm leading-normal">
              <th className="py-3 px-6 text-left">Symbol</th>
              <th className="py-3 px-6 text-left">Quantity</th>
              <th className="py-3 px-6 text-left">Price</th>
              <th className="py-3 px-6 text-left">Status</th>
            </tr>
          </thead>
          <tbody className="text-gray-600 text-sm font-light">
            {trades.map((trade) => (
              <tr
                key={trade.id}
                className="border-b border-gray-200 hover:bg-gray-100"
              >
                <td className="py-3 px-6 text-left whitespace-nowrap font-medium">{trade.symbol}</td>
                <td className="py-3 px-6 text-left">{trade.quantity}</td>
                <td className="py-3 px-6 text-left">${trade.price.toFixed(2)}</td>
                <td
                  className={`py-3 px-6 text-left font-semibold ${
                    trade.status === "Open" ? "text-green-600" : "text-red-600"
                  }`}
                >
                  {trade.status}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default App;

