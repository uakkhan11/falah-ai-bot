import { useState, useEffect } from "react";
import PriceChart from "./PriceChart";

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [trades, setTrades] = useState([]);
  const [portfolio, setPortfolio] = useState(null);
  const [filter, setFilter] = useState("All");
  const [sortOrder, setSortOrder] = useState("desc");

  // Fetch portfolio and trades data from backend API
  useEffect(() => {
    async function fetchPortfolio() {
      try {
        const response = await fetch("http://168.231.123.222:8000/api/portfolio");
        const data = await response.json();
        setPortfolio(data);
      } catch (error) {
        console.error("Error fetching portfolio:", error);
      }
    }
    async function fetchTrades() {
      try {
        const response = await fetch("http://168.231.123.222:8000/api/trades");
        const data = await response.json();
        setTrades(data);
      } catch (error) {
        console.error("Error fetching trades:", error);
      }
    }
    fetchPortfolio();
    fetchTrades();

    // Optional: refresh data every 15 seconds
    const interval = setInterval(() => {
      fetchPortfolio();
      fetchTrades();
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  // Filter trades by status
  const filteredTrades =
    filter === "All"
      ? trades
      : trades.filter((trade) => trade.status === filter);

  // Sort trades by price
  const sortedTrades = [...filteredTrades].sort((a, b) =>
    sortOrder === "asc" ? a.price - b.price : b.price - a.price
  );

  return (
    <>
      <div className="bg-blue-600 text-white p-4 font-bold">
        Tailwind CSS is working!
      </div>
      <div className="flex min-h-screen bg-gray-100">
        {/* Sidebar */}
        <aside
          className={`bg-gray-900 text-white ${
            sidebarOpen ? "w-64" : "w-16"
          } transition-all duration-200 flex flex-col`}
        >
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <h2 className={`text-xl font-bold ${sidebarOpen ? "block" : "hidden"}`}>
              Falah Dashboard
            </h2>
            <button
              className="text-white text-2xl focus:outline-none"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle Sidebar"
            >
              ☰
            </button>
          </div>
          <nav className="flex-1 p-2 space-y-2">
            <a
              href="#"
              className="block py-2 px-3 rounded hover:bg-gray-700 transition-colors"
              title="Home"
            >
              🏠 {sidebarOpen && "Home"}
            </a>
            <a
              href="#"
              className="block py-2 px-3 rounded hover:bg-gray-700 transition-colors"
              title="Trades"
            >
              📈 {sidebarOpen && "Trades"}
            </a>
            <a
              href="#"
              className="block py-2 px-3 rounded hover:bg-gray-700 transition-colors"
              title="Live"
            >
              🔴 {sidebarOpen && "Live"}
            </a>
            <a
              href="#"
              className="block py-2 px-3 rounded hover:bg-gray-700 transition-colors"
              title="Settings"
            >
              ⚙️ {sidebarOpen && "Settings"}
            </a>
          </nav>
        </aside>
        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <header className="bg-white shadow flex items-center justify-between px-6 py-4">
            <h1 className="text-2xl font-semibold">Falah AI Bot Dashboard</h1>
            <div>User</div>
          </header>
          <main className="flex-1 p-6 overflow-auto">
            {/* Dashboard Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-8">
              <div className="bg-white rounded-lg shadow-md p-6 text-center">
                <div className="text-gray-500 mb-2">Portfolio Value</div>
                <div className="text-3xl font-bold text-blue-600">
                  {portfolio ? `$${portfolio.portfolio_value}` : "Loading..."}
                </div>
              </div>
              <div className="bg-white rounded-lg shadow-md p-6 text-center">
                <div className="text-gray-500 mb-2">Today's Profit</div>
                <div className="text-3xl font-bold text-green-600">
                  {portfolio ? portfolio.todays_profit : "Loading..."}
                </div>
              </div>
              <div className="bg-white rounded-lg shadow-md p-6 text-center">
                <div className="text-gray-500 mb-2">Open Trades</div>
                <div className="text-3xl font-bold text-purple-600">
                  {portfolio ? portfolio.open_trades : "Loading..."}
                </div>
              </div>
            </div>
            {/* Portfolio Chart */}
            <PriceChart />
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
                            trade.status === "Open"
                              ? "text-green-600"
                              : "text-red-600"
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
          </main>
        </div>
      </div>
    </>
  );
}

export default App;
