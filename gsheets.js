// gsheets.js
(function(){
  // <-- Paste your published Apps Script URL here
  const SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbxklIImFWEAgFFLHLjFVBPI5mMP2cTUkkKAhX2FNj95CVCi716aLUmTN_kpInu0h1dAeg/exec';

  function logToSheet(data) {
    return new Promise((resolve, reject) => {
      const callbackName = 'gsheet_cb_' + Date.now();
      window[callbackName] = function(response) {
        delete window[callbackName];
        document.body.removeChild(scriptTag);
        console.log('Sheets JSONP callback:', response);
        resolve(response);
      };

      const params = new URLSearchParams(data);
      params.append('callback', callbackName);

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

  // Expose it globally so your inline buttons and tradeLogic.js can see it:
  window.logToSheet = logToSheet;
})();
