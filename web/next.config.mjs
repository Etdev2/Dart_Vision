/**
 * Static export.
 *
 * #4 decided there is no server at any tier: inference runs on the device, and
 * the app is a PWA over static assets on a CDN. `output: 'export'` makes that
 * the build's actual shape rather than an intention -- anything needing a
 * server fails the build instead of quietly requiring one in production.
 *
 * @type {import('next').NextConfig}
 */
export default {
  output: 'export',
  // Without this the export writes `match.html`, while the nav links point at
  // `/match/` -- so a refresh or a shared deep link serves a directory, not the
  // page. Verified against the built output rather than assumed: the links and
  // the emitted file names have to agree.
  trailingSlash: true,
  images: { unoptimized: true },
  // The decision logic in lib/ is plain ES modules with JSDoc types, shared
  // unchanged with the no-build demo in public/standalone.html.
  eslint: { ignoreDuringBuilds: true },
};
