'use client';

/**
 * Registers the service worker, and keeps the screen awake during a match.
 *
 * The wake lock is the other half of #4's open question. That record named
 * "can a browser hold a camera open for a fifteen-minute leg -- screen sleep,
 * wake lock, backgrounding" as the one finding that would force a native
 * wrapper. This is the browser's answer to the first two, and it costs nothing
 * to take: the lock is released automatically when the page is hidden, and
 * re-taken when it comes back.
 */

import { useEffect } from 'react';

export default function OfflineReady() {
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js').catch(() => {
        // No offline support. Worth nothing to the player to be told.
      });
    }

    const wakeLock = { current: null as WakeLockSentinel | null };
    const request = async () => {
      try {
        if (document.visibilityState === 'visible' && 'wakeLock' in navigator) {
          wakeLock.current = await navigator.wakeLock.request('screen');
        }
      } catch {
        // Denied, unsupported, or the battery is low. The app still works.
      }
    };
    request();
    document.addEventListener('visibilitychange', request);
    return () => {
      document.removeEventListener('visibilitychange', request);
      wakeLock.current?.release().catch(() => {});
    };
  }, []);

  return null;
}
