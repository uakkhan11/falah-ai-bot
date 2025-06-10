// tradeLogic.js
window.runTradingCycle = async function() {
  const cfg = JSON.parse(localStorage.getItem('falahConfig') || '{}');
  console.log('Config:', cfg);

  // 1) Ensure we have an access token
  if (cfg.brokerName === 'Zerodha') {
    if (!cfg.accessToken && cfg.requestToken) {
      cfg.accessToken = (await exchangeToken(cfg.requestToken)).data.access_token;
      // Persist back
      localStorage.setItem('falahConfig', JSON.stringify(cfg));
    }
  }

  // 2) Build your signal
  const signal = { stock:'TCS', entry:3700, exit:3745, pl:45, reason:'Auto Trade' };

  // 3) Place the order via proxy
  try {
    const orderResp = await placeOrder(cfg.brokerName, {
      stock: signal.stock,
      entry: signal.entry,
      access_token: cfg.accessToken
    });
    console.log('Order response:', orderResp);

    // 4) Notify & log
    sendTelegramMessage(`✅ Traded ${signal.stock}@₹${signal.entry}`);
    logToCSV(signal);
  } catch (err) {
    console.error('Trade failed:', err);
    sendTelegramMessage(`❌ Trade failed: ${err.message}`);
  }
};
