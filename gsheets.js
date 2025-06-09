const SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbwaFCbbNhf4VpLLVufLvCwTyprC45k26gBmr_A4C7c26lqq7-K4sQT5TUAiMTfiEVYHkg/exec';
async function logToSheet(trade) {
  let res = await fetch(SCRIPT_URL, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(trade)
  });
  console.log('Sheets:', await res.text());
}
