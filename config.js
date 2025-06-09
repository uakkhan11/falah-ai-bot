const CONFIG_KEY = 'falahConfig';
const DEFAULTS = {
  capital: 20000,
  maxTrades: 5,
  brokerName: 'AngelOne'
};

// Load config or fall back to defaults
function loadConfig() {
  const raw = localStorage.getItem(CONFIG_KEY);
  return raw ? JSON.parse(raw) : DEFAULTS;
}

// Save config back to localStorage
function saveConfig(cfg) {
  localStorage.setItem(CONFIG_KEY, JSON.stringify(cfg));
}

const CONFIG_KEY = 'falahConfig';
const DEFAULTS = { capital:20000, maxTrades:5, brokerName:'AngelOne' };
window.loadConfig = () => JSON.parse(localStorage.getItem(CONFIG_KEY) || JSON.stringify(DEFAULTS));
