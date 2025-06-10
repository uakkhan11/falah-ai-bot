// broker.js
const PROXY_URL = 'https://falah-proxy.uakkhan11.workers.dev/';

async function proxyRequest(body) {
  const res = await fetch(PROXY_URL, {
    method: 'POST',
    headers:{ 'Content-Type':'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error('Network error ' + res.status);
  return res.json();
}

async function exchangeToken(requestToken) {
  return proxyRequest({ request_token: requestToken });
}

async function placeOrder(broker, signal) {
  const cfg = loadConfig().credentials[broker];
  if (broker === 'Zerodha') {
    const token = cfg.accessToken;    // use the pasted token
    const res = await fetch('https://api.kite.trade/orders/regular', {
      method: 'POST',
      headers: {
        'Content-Type':'application/json',
        'X-Kite-Version':'3',
        'Authorization': `token ${cfg.apiKey}:${token}`
      },
      body: JSON.stringify({
        exchange: 'NSE',
        tradingsymbol: signal.stock,
        transaction_type: 'BUY',
        quantity: 1,
        product: 'CNC',
        order_type: 'MARKET'
      })
    });
    return res.json();
  }
  throw new Error('Unsupported broker');
}
window.placeOrder = placeOrder;
