// config.js

// A single global key for localStorage
window.CONFIG_KEY = 'falahConfig';

// Default settings
window.DEFAULTS = {
  capital:    20000,
  maxTrades:  5,
  brokerName: 'AngelOne'
};

// Expose this function globally
window.loadConfig = function() {
  try {
    return JSON.parse(localStorage.getItem(window.CONFIG_KEY)) || window.DEFAULTS;
  } catch {
    return window.DEFAULTS;
  }
};
