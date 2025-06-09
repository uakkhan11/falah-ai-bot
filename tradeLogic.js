// tradeLogic.js
(function(){
  let todayTrades = 0;
  let todayUsedCapital = 0;

  // Load settings from config.js
  function loadConfig() {
    try {
      return JSON.parse(localStorage.getItem('falahConfig')) || {};
    } catch {
      return {};
    }
  }

  // Mock placeOrder
  async function placeOrder(broker, signal) {
    // Simulate network latency
    await new Promise(r => setTimeout(r, 200));
    return { broker, ...signal, orderId: Date.now() };
  }

  // The main entrypoint
  async function runTradingCycle() {
    const cfg = loadConfig();
    console.log(`Config → capital:₹${cfg.capital}, maxTrades:${cfg.maxTrades}, broker:${cfg.brokerName}`);

    // Dummy signal
    const signal = { stock:'TCS', entry:3700, exit:3745, pl:45, reason:'Auto Trade' };

    if (todayTrades >= cfg.maxTrades) {
      console.warn('🍂 Max trades reached for today.');
      return;
    }
    if ((todayUsedCapital + signal.entry) > cfg.capital) {
      console.warn('🚫 Not enough capital.');
      return;
    }

    try {
      const res = await placeOrder(cfg.brokerName, signal);
      console.log('🚀 Order placed:', res);
      todayTrades++;
      todayUsedCapital += signal.entry;

      // Telegram
      sendTelegramMessage(`✅ Traded ${signal.stock} @ ₹${signal.entry} via ${cfg.brokerName}`);

      // CSV
      if (typeof logToCSV === 'function') {
        logToCSV(signal);
        console.log('💾 Logged to CSV buffer:', signal);
      } else {
        console.warn('⚠️ logToCSV not defined');
      }
    } catch (err) {
      console.error('❌ Trade failed:', err);
      sendTelegramMessage(`❌ Trade failed: ${err.message}`);
    }
  }

  window.runTradingCycle = runTradingCycle;
})();
