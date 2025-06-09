// JSONP-based sheet logger
function logToSheet(trade) {
  const base = 'https://script.google.com/macros/s/AKfycbw-n4J-2WsGS9c5nwVckdcZ65XwhP2NMrbsNf0B28p9_Cu_MeHBD4fuTmMjRXDrDdWhKQ/exec'; 
  const params = new URLSearchParams({
    stock:    trade.stock,
    entry:    trade.entry,
    exit:     trade.exit,
    pl:       trade.pl,
    reason:   trade.reason,
    callback: 'onSheetResponse'
  });
  const s = document.createElement('script');
  s.src = `${base}?${params}`;
  document.body.appendChild(s);
}

// Called by the JSONP response
function onSheetResponse(resp) {
  console.log('Sheets JSONP:', resp);
  alert('Logged to sheet: ' + resp.status);
}
