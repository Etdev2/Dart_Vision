/**
 * Offline, because dartboards live in garages.
 *
 * #4 decided there is no server at any tier, which makes offline a matter of
 * caching rather than of architecture -- and #2's mounting band puts the phone
 * in a shed or a basement, where connectivity is the normal problem rather
 * than the edge case. A scorer that dies when the wifi dips mid-leg is not a
 * scorer.
 *
 * Written by hand rather than pulled from a plugin. It is forty lines, a
 * dependency would ship a third party's code into the offline path, and the
 * caching strategy here is worth being able to read.
 *
 * **Cache-first for assets, network-first for navigations.** Next emits
 * content-hashed filenames, so an asset URL's content never changes and
 * cache-first is exact rather than merely fast. Navigations go to the network
 * first so a deployed update is picked up on the next load, and fall back to
 * the cached shell when there is nothing to reach.
 */

const VERSION = 'dartvision-v1';
const SHELL = ['/', '/match/', '/manifest.webmanifest', '/icon-192.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(VERSION)
      // One missing entry must not fail the whole install, or a renamed route
      // leaves the app with no offline support at all and no sign of why.
      .then((cache) => Promise.allSettled(SHELL.map((url) => cache.add(url))))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(VERSION).then((cache) => cache.put(request, copy));
          return response;
        })
        .catch(async () =>
          (await caches.match(request)) ?? (await caches.match('/')) ?? Response.error()),
    );
    return;
  }

  event.respondWith(
    caches.match(request).then((hit) => hit ?? fetch(request).then((response) => {
      if (response.ok) {
        const copy = response.clone();
        caches.open(VERSION).then((cache) => cache.put(request, copy));
      }
      return response;
    })),
  );
});
