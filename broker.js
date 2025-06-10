// broker.js
const PROXY_URL = 'https://script.google.com/macros/s/AKfycbyiWSQ1f2TvFeMcEPQxA8HWaJxcMVuZEOfmVGaAEbEo_qDn20pChrvD6mhB9tRWCgV1Og/exec';

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
  const cfg = JSON.parse(localStorage.getItem('falahConfig')||'{}');
  const token = cfg.accessToken || (await exchangeToken(cfg.requestToken)).data.access_token;
  // Save access token
  cfg.accessToken = token;
  localStorage.setItem('falahConfig', JSON.stringify(cfg));

  // Place order
  return proxyRequest({ broker, ...signal, access_token: token });
}

window.exchangeToken = exchangeToken;
window.placeOrder    = placeOrder;
