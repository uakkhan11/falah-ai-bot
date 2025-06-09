// tradeLogic.js
(function(){
  // these vars are private and only declared once per page load
  let todayTrades      = 0;
  let todayUsedCapital = 0;

  async function runTradingCycle() {
    const cfg = loadConfig();
    console.log(`Configured Capital: ₹${cfg.capital}, Max Trades: ${cfg.maxTrades}, Broker: ${cfg.brokerName}`);

    // Example signal
    const signal = { stock:'TCS', entry:3700, exit:3745, pl:45, reason:'Auto Trade' };

    // Check limits
    if (todayTrades >= cfg.maxTrades) {
      console.warn('Max trades reached.');
      return;
    }
    if (todayUsedCapital + signal.entry > cfg.capital) {
      console.warn('Not enough capital.');
      return;
    }

    // “Place” the order
    const response = await placeOrder(cfg.brokerName, signal);
    console.log('Order response:', response);

    // Update counters
    todayTrades++;
    todayUsedCapital += signal.entry;

    // Telegram
    sendTelegramMessage(`✅ Traded ${signal.stock} @ ${signal.entry}`);

    // CSV logging
    logToCSV(signal);
  }

  // Export globally
  window.runTradingCycle = runTradingCycle;

  // Mock placeOrder
  window.placeOrder = (broker, signal) =>
    Promise.resolve({ broker, ...signal, orderId:Date.now() });
})();
