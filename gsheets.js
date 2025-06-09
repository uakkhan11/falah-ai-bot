const SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbxibhA8nKQ50gtxPTjib_3MRUihb-_LZKM5BplA5Z2s43fqIfjBj3saDWNt9rkLuZaf1w/exec';
async function logToSheet(trade) {
  let res = await fetch(SCRIPT_URL, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(trade)
  });
  console.log('Sheets:', await res.text());
}
