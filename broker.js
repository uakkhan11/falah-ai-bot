// broker.js
const PROXY_URL = 'https://script.google.com/macros/s/AKfycbwHcqFg5z1ICvrtjMXZpBuQ-t_6v91maQSJ4r92mH1-jix1mXqxnIvB_Kf5_poArsQaEA/exec';

async function proxyRequest(body) {
  const res = await fetch(PROXY_URL, {
    method:  'POST',
    headers: { 'Content-Type':'application/json' },
    body:    JSON.stringify(body)
  });
  return res.json();
}

// Exchange request_token for access_token
async function exchangeToken(requestToken) {
  const resp = await proxyRequest({ request_token: requestToken });
  if (resp.status === 'error') throw new Error(resp.message);
  return resp.data.access_token;
}

// Place a real order
async function placeOrder(broker, signal) {
  const cfg = loadConfig();
  if (broker === 'Zerodha') {
    const token = cfg.accessToken || await exchangeToken(cfg.requestToken);
    // Save it for reuse
    localStorage.setItem('falahAccessToken', token);

    const resp = await proxyRequest({
      broker,
      stock: signal.stock,
      entry: signal.entry,
      access_token: token
    });
    if (!resp.status && resp.status !== 'success') {
      throw new Error(resp.message || 'Order failed');
    }
    return resp.data || resp;
  }
  throw new Error('Unsupported broker: ' + broker);
}

window.placeOrder     = placeOrder;
window.exchangeToken  = exchangeToken;
