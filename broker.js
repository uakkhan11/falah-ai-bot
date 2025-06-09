// broker.js

// The URL of your proxy endpoint (replace with your deployment URL)
const BROKER_PROXY = 'https://your-proxy.example.com/executeOrder';

async function placeOrder(broker, { stock, entry }) {
  // Send through your proxy so you don't expose keys in the browser
  const res = await fetch(BROKER_PROXY, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ broker, stock, entry })
  });
  if (!res.ok) throw new Error(`Broker API error ${res.status}`);
  return res.json();
}

// Expose globally
window.placeOrder = placeOrder;
