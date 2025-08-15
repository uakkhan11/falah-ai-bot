import { useState } from "react";

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <aside
        className={`bg-gray-900 text-white transition-all duration-300 ${
          sidebarOpen ? "w-64" : "w-16"
        } flex flex-col`}
      >
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className={`text-xl font-bold ${sidebarOpen ? "block" : "hidden"}`}>
            Falah Dashboard
          </h2>
          <button
            className="text-white text-2xl focus:outline-none"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle sidebar"
          >
            &#9776;
          </button>
        </div>
        <nav className="flex-1 p-2 space-y-2">
          <a
            href="#"
            className="block rounded py-2 px-3 hover:bg-gray-700 transition-colors"
            title="Home"
          >
            🏠 {sidebarOpen && "Home"}
          </a>
          <a
            href="#"
            className="block rounded py-2 px-3 hover:bg-gray-700 transition-colors"
            title="Trades"
          >
            📈 {sidebarOpen && "Trades"}
          </a>
          <a
            href="#"
            className="block rounded py-2 px-3 hover:bg-gray-700 transition-colors"
            title="Live"
          >
            🔴 {sidebarOpen && "Live"}
          </a>
          <a
            href="#"
            className="block rounded py-2 px-3 hover:bg-gray-700 transition-colors"
            title="Settings"
          >
            ⚙️ {sidebarOpen && "Settings"}
          </a>
        </nav>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col">
        <header className="bg-white shadow flex items-center justify-between px-6 py-4">
          <h1 className="text-2xl font-semibold">Dashboard</h1>
          <div>User</div>
        </header>

        <main className="flex-1 p-6 overflow-auto">
          <h2 className="text-xl font-bold mb-4">Welcome to Falah AI Bot Dashboard</h2>
          {/* Future dashboard content goes here */}
        </main>
      </div>
    </div>
  );
}

export default App; 
