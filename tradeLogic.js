// tradeLogic.js

// In-memory counters for today's trades
let todayTrades       = 0;
let todayUsedCapital  = 0;

async function runTradingCycle() {
  const cfg = window.loadConfig();
  console.log(`Configured Capital: ₹${cfg.capital}, Max Trades: ${cfg.maxTrades}, Broker: ${cfg.brokerName}`);

  // Example signal
  const signal = { stock:'TCS', entry:3700, exit:3745, pl:45, reason:'Auto Trade' };

  // Enforce maxTrades
  if (todayTrades >= cfg.maxTrades) {
    console.warn('🍂 Max trades reached for today.');
    return;
  }

  // Enforce capital
  if (todayUsedCapital + signal.entry > cfg.capital) {
    console.warn('🚫 Not enough capital for this trade.');
    return;
  }

  try {
    // Simulate order placement
    const response = await placeOrder(cfg.brokerName, signal);
    console.log('💹 Order placed successfully:', response);

    todayTrades++;
    todayUsedCapital += signal.entry;

    // Alert via Telegram
    sendTelegramMessage(`✅ Traded ${signal.stock} at ₹${signal.entry} via ${cfg.brokerName}`);

    // Log to CSV (replacing Sheets)
    if (typeof logToCSV === 'function') {
      logToCSV(signal);
    }

  } catch (err) {
    console.error('❌ Trade failed:', err);
    sendTelegramMessage(`❌ Trade failed on ${signal.stock}: ${err.message}`);
  }
}

// Mock broker connectors
async function placeOrder(broker, { stock, entry }) {
  // Replace these stubs with real API calls later
  return Promise.resolve({ broker, stock, entry, orderId:Date.now() });
}

// Expose for console & button use
window.runTradingCycle = runTradingCycle;
