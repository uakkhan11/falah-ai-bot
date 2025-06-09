self.addEventListener('install', e => {
  e.waitUntil(
    caches.open('falah').then(c =>
      c.addAll(['index.html','telegram.js','gsheets.js','manifest.json'])
    )
  );
});
self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request))
  );
});
