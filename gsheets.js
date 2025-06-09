self.addEventListener('install', e => {
  e.waitUntil(
    caches.open('falah').then(cache =>
      cache.addAll([
        'index.html',
        'telegram.js',
        'gsheets.js',
        'manifest.json',
        'icon.png'
      ])
    )
  );
});

self.addEventListener('fetch', e => {
  const reqUrl = new URL(e.request.url);

  // If it’s not from our origin, skip the worker and let the browser handle it.
  if (reqUrl.origin !== location.origin) {
    return;
  }

  // Otherwise serve from cache, fall back to network
  e.respondWith(
    caches.match(e.request).then(cached =>
      cached || fetch(e.request)
    )
  );
});
