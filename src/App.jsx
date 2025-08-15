import { useState } from "react";

function App() {
  // Sample trades data and UI states
  const initialTrades = [
    { id: 1, symbol: "AAPL", quantity: 10, price: 192.38, status: "Open" },
    { id: 2, symbol: "TSLA", quantity: 5, price: 247.11, status: "Closed" },
    { id: 3, symbol: "BTCUSD", quantity: 0.15, price: 31000.5, status: "Open" },
    { id: 4, symbol: "GOOG", quantity: 2, price: 132.11, status: "Closed" },
  ];

  const [filter, setFilter] = useState("All");
  const [sortOrder, setSortOrder] = useState("desc");

  const filteredTrades =
    filter === "All"
      ? initialTrades
      : initialTrades.filter((trade) => trade.status === filter);

  const sortedTrades = [...filteredTrades].sort((a, b) => {
    if (sortOrder === "asc") return a.price - b.price;
    else return b.price - a.price;
  });

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      {/* Dashboard Header */}
      <h1 className="text-4xl font-bold text-blue-600 mb-6">Falah AI Bot Dashboard</h1>

      {/* Dashboard Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-gray-500 mb-2">Portfolio Value</div>
          <div className="text-3xl font-bold text-blue-600">$25,421</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-gray-500 mb-2">Today's Profit</div>
          <div className="text-3xl font-bold text-green-600">+7.2%</div>
        </div>
        <div className="bg-white rounded-lg shadow-md p-6 text-center">
          <div className="text-gray-500 mb-2">Open Trades</div>
          <div className="text-3xl font-bold text-purple-600">12</div>
        </div>
      </div>

      {/* Trades Table Section */}
      <div>
        <h2 className="text-2xl font-semibold mb-4">Trades Table</h2>

        {/* Filter and Sort Controls */}
        <div className="mb-4 flex flex-wrap items-center gap-4">
          <div>
            <label className="mr-2 font-semibold">Filter Status:</label>
            <select
              className="p-2 border rounded"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            >
              <option value="All">All</option>
              <option value="Open">Open</option>
              <option value="Closed">Closed</option>
            </select>
          </div>
          <div>
            <label className="mr-2 font-semibold">Sort Price:</label>
            <button
              className="bg-blue-600 text-white px-3 py-2 rounded mr-2"
              onClick={() => setSortOrder("asc")}
            >
              Low → High
            </button>
            <button
              className="bg-blue-600 text-white px-3 py-2 rounded"
              onClick={() => setSortOrder("desc")}
            >
              High → Low
            </button>
          </div>
        </div>

        {/* Trades Table */}
        <div className="overflow-x-auto bg-white rounded shadow p-4">
          <table className="min-w-full table-auto">
            <thead>
              <tr className="bg-gray-200 text-gray-600 uppercase text-sm leading-normal">
                <th className="py-3 px-6 text-left">Symbol</th>
                <th className="py-3 px-6 text-left">Quantity</th>
                <th className="py-3 px-6 text-left">Price</th>
                <th className="py-3 px-6 text-left">Status</th>
                <th className="py-3 px-6 text-left">Action</th>
              </tr>
            </thead>
            <tbody className="text-gray-600 text-sm font-light">
              {sortedTrades.map((trade) => (
                <tr
                  key={trade.id}
                  className="border-b border-gray-200 hover:bg-gray-100"
                >
                  <td className="py-3 px-6 font-medium">{trade.symbol}</td>
                  <td className="py-3 px-6">{trade.quantity}</td>
                  <td className="py-3 px-6">${trade.price.toFixed(2)}</td>
                  <td
                    className={`py-3 px-6 font-semibold ${
                      trade.status === "Open" ? "text-green-600" : "text-red-600"
                    }`}
                  >
                    {trade.status}
                  </td>
                  <td className="py-3 px-6">
                    {trade.status === "Open" && (
                      <button
                        className="bg-red-500 text-white px-3 py-1 rounded text-xs"
                        onClick={() => alert(`Closing trade for ${trade.symbol}`)}
                      >
                        Close
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default App;
