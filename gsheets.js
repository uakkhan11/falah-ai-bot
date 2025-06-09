// gsheets.js

(function(){
  const SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbw…/exec';  // your published Apps Script URL

  function logToSheet(data) {
    // Build JSONP callback name
    const callbackName = 'gsheet_cb_' + Date.now();
    return new Promise((resolve, reject) => {
      window[callbackName] = function(response) {
        // Clean up
        delete window[callbackName];
        document.body.removeChild(scriptTag);
        resolve(response);
      };

      // Append the JSONP <script> tag
      const params = new URLSearchParams(data);
      params.append('callback', callbackName);
      const scriptTag = document.createElement('script');
      scriptTag.src = `${SCRIPT_URL}?${params.toString()}`;
      scriptTag.onerror = () => {
        delete window[callbackName];
        document.body.removeChild(scriptTag);
        reject(new Error('JSONP request failed'));
      };
      document.body.appendChild(scriptTag);
    });
  }

  // Export globally
  window.logToSheet = logToSheet;
})();
