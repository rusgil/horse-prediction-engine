// Vercel Edge Middleware — staging-only HTTP Basic Auth gate.
//
// STAGING-ONLY FILE: do NOT merge into main. It keeps the public off the
// staging site — which protects the AdSense account from invalid-traffic /
// duplicate-content flags on a clone, keeps the staging URL out of search
// indexes, and blocks stray access to the (weak-secret) staging backend.
//
// It is a NO-OP unless the STAGING_PASSWORD env var is set, so even if this
// file ever reached prod it would do nothing there. Set STAGING_PASSWORD (and
// optionally STAGING_USER, default "staging") on the STAGING Vercel project
// only. Gates every route, including the /api/* rewrite (middleware runs before
// rewrites), so both the pages and the proxied API require the password.

export const config = { matcher: '/:path*' };

export default function middleware(request) {
  const pass = process.env.STAGING_PASSWORD;
  if (!pass) return; // not configured (e.g. prod) → let everything through

  const user = process.env.STAGING_USER || 'staging';
  const header = request.headers.get('authorization') || '';
  if (header.startsWith('Basic ')) {
    try {
      const decoded = atob(header.slice(6));
      const idx = decoded.indexOf(':');
      const u = decoded.slice(0, idx);
      const p = decoded.slice(idx + 1);
      if (u === user && p === pass) return; // authenticated → continue
    } catch (_) {
      // malformed header → fall through to 401
    }
  }

  return new Response('FunkyIQ staging — authentication required.', {
    status: 401,
    headers: {
      'WWW-Authenticate': 'Basic realm="FunkyIQ Staging", charset="UTF-8"',
      'X-Robots-Tag': 'noindex, nofollow',
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-store',
    },
  });
}
