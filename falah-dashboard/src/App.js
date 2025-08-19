import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import icon512 from './assets/icon-512.jpeg';
import PriceChart from "./PriceChart";
import OpenTrades from "./components/OpenTrades";
import ManualTrade from "./components/ManualTrade";
import ManualExit from "./components/ManualExit";
import './App.css'

function Home() {
  return <div>Welcome to the Home page!</div>;
}
function Live() {
  return <div>Live Status Page</div>;
}
function Settings() {
  return <div>Settings Page</div>;
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [trades, setTrades] = useState([]);
  const [portfolio, setPortfolio] = useState(null);
  const [filter, setFilter] = useState("All");
  const [sortOrder, setSortOrder] = useState("desc");

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
    const interval = setInterval(() => {
      fetchPortfolio();
      fetchTrades();
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const filteredTrades =
    filter === "All"
      ? trades
      : trades.filter((trade) => trade.status === filter);
  const sortedTrades = [...filteredTrades].sort((a, b) =>
    sortOrder === "asc" ? a.price - b.price : b.price - a.price
  );

  return (
    <Router>
      {/* Outer background container */}
      <div className="bg-dashboard min-h-screen relative">
        {/* Overlay */}
        <div className="bg-overlay fixed inset-0 z-10"></div>

        {/* Main app container */}
        <div className="main-content relative z-20 flex min-h-screen bg-gray-100">
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
              <NavLink
                to="/"
                className={({ isActive }) =>
                  `block py-2 px-3 rounded hover:bg-gray-700 transition-colors ${
                    isActive ? "bg-blue-700 font-bold" : ""
                  }`
                }
                title="Home"
              >
                🏠 {sidebarOpen && "Home"}
              </NavLink>
              {/* Repeat NavLinks for other routes (Trades, Manual Trade, etc.) */}
              <NavLink
                to="/trades"
                className={({ isActive }) =>
                  `block py-2 px-3 rounded hover:bg-gray-700 transition-colors ${
                    isActive ? "bg-blue-700 font-bold" : ""
                  }`
                }
                title="Trades"
              >
                📈 {sidebarOpen && "Trades"}
              </NavLink>
              <NavLink
                to="/manual-trade"
                className={({ isActive }) =>
                  `block py-2 px-3 rounded hover:bg-gray-700 transition-colors ${
                    isActive ? "bg-blue-700 font-bold" : ""
                  }`
                }
                title="Manual Trade"
              >
                💰 {sidebarOpen && "Manual Trade"}
              </NavLink>              
              <NavLink
                to="/manual-exit"
                className={({ isActive }) =>
                  `block py-2 px-3 rounded hover:bg-gray-700 transition-colors ${
                    isActive ? "bg-blue-700 font-bold" : ""
                  }`
                }
                title="Manual Exit"
              >
                🛑 {sidebarOpen && "Manual Exit"}
              </NavLink>
              <NavLink
                to="/live"
                className={({ isActive }) =>
                  `block py-2 px-3 rounded hover:bg-gray-700 transition-colors ${
                    isActive ? "bg-blue-700 font-bold" : ""
                  }`
                }
                title="Live"
              >
                🔴 {sidebarOpen && "Live"}
              </NavLink>
              <NavLink
                to="/settings"
                className={({ isActive }) =>
                  `block py-2 px-3 rounded hover:bg-gray-700 transition-colors ${
                    isActive ? "bg-blue-700 font-bold" : ""
                  }`
                }
                title="Settings"
              >
                ⚙️ {sidebarOpen && "Settings"}
              </NavLink>
            </nav>
          </aside>

          {/* Main Content Area */}
          <div className="flex-1 flex flex-col">
            {/* Header */}
            <header className="bg-white shadow flex items-center justify-between px-6 py-4">
              <img src={icon512} alt="Falah Logo" className="logo mx-auto max-w-[220px]" />
              <h1 className="text-2xl font-semibold">Falah AI Bot Dashboard</h1>
              <div>User</div>
            </header>

            {/* Main page content */}
            <main className="flex-1 p-6 overflow-auto">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/trades" element={<OpenTrades />} />
                <Route path="/manual-trade" element={<ManualTrade />} />
                <Route path="/manual-exit" element={<ManualExit />} />
                <Route path="/live" element={<Live />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </main>
          </div>
        </div>
      </div>
    </Router>
  );
}

export default App;
