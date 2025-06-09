// gsheets.js
(function() {
  // Replace with your Apps Script Web App URL (deployed as "Anyone, even anonymous")
  const SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbxklIImFWEAgFFLHLjFVBPI5mMP2cTUkkKAhX2FNj95CVCi716aLUmTN_kpInu0h1dAeg/exec';

  /**
   * Append a row to Google Sheets via JSONP
   * @param {{ stock: string, entry: number, exit: number, pl: number, reason: string }} data
   * @returns {Promise<object>} The JSONP callback response
   */
  function logToSheet(data) {
    return new Promise((resolve, reject) => {
      // unique callback name
      const callbackName = 'gsheet_cb_' + Date.now();
      // define the global callback
      window[callbackName] = function(response) {
        // Cleanup
        delete window[callbackName];
        document.body.removeChild(scriptTag);
        resolve(response);
      };

      // build query string
      const params = new URLSearchParams(data);
      params.append('callback', callbackName);

      // inject JSONP script
      const scriptTag = document.createElement('script');
      scriptTag.src = SCRIPT_URL + '?' + params.toString();
      scriptTag.onerror = () => {
        delete window[callbackName];
        document.body.removeChild(scriptTag);
        reject(new Error('JSONP request failed'));
      };
      document.body.appendChild(scriptTag);
    });
  }

  // expose globally
  window.logToSheet = logToSheet;
})();
