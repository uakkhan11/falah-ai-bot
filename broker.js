async function placeOrder(broker, {stock, entry}) {
  const cfg = loadConfig();
  const creds = cfg.credentials[broker];
  // Use creds.apiKey and creds.apiSecret to authenticate your broker API request
  // E.g., for Zerodha, include them in headers/token
  return { success:true, broker, stock, entry, orderId:Date.now() };
}
window.placeOrder = placeOrder;
