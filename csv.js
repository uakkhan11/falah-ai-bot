// csv.js
(function(){
  let _lines = [];

  // Call this in your tradeLogic when a trade executes
  function logToCSV(trade) {
    const { stock, entry, exit, pl, reason } = trade;
    const timestamp = new Date().toISOString();
    const row = [stock, entry, exit, pl, reason, timestamp]
      .map(cell => `"${String(cell).replace(/"/g,'""')}"`)
      .join(',');
    _lines.push(row);
  }

  // Call this to download at any time
  function downloadCSV() {
    if (!_lines.length) {
      alert('No trades logged yet!');
      return;
    }
    const header = ['Stock','Entry','Exit','P/L','Reason','Time'].join(',');
    const csv   = [header, ..._lines].join('\r\n');
    const blob  = new Blob([csv], { type: 'text/csv' });
    const url   = URL.createObjectURL(blob);
    const a     = document.createElement('a');
    a.href      = url;
    a.download  = 'trades.csv';
    document.body.appendChild(a);  // required for Firefox
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // Export globally
  window.logToCSV     = logToCSV;
  window.downloadCSV  = downloadCSV;
})();
