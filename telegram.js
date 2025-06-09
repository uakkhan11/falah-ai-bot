async function sendTelegramMessage(text) {
  const botToken = '7763450358:AAH32bWYyu_hXR6l-UaVMaarFGZ4YFOv6q8';
  const chatId   = '6784139148';
  const url = `https://api.telegram.org/bot${botToken}/sendMessage`;
  let res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ chat_id: chatId, text })
  });
  console.log('Telegram:', await res.json());
}
