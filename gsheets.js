function logToSheet(trade) {
  const baseUrl = 'https://script.google.com/macros/s/AKfycbw-n4J-2WsGS9c5nwVckdcZ65XwhP2NMrbsNf0B28p9_Cu_MeHBD4fuTmMjRXDrDdWhKQ/exec'; 
    // ← REPLACE with your actual Web app URL

  // Build query string with JSONP callback
  const params = new URLSearchParams({
    stock:   trade.stock,
    entry:   trade.entry,
    exit:    trade.exit,
    pl:      trade.pl,
    reason:  trade.reason,
    callback:'onSheetResponse'
  });

  // Inject a <script> tag to perform the JSONP request
  const script = document.createElement('script');
  script.src = `${baseUrl}?${params}`;
  document.body.appendChild(script);
}

// JSONP callback invoked by Apps Script
function onSheetResponse(response) {
  console.log('Sheets JSONP response:', response);
  alert('Logged to sheet: ' + response.status);
}
