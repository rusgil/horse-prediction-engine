export default function middleware(req) {
  const url = new URL(req.url);
  const path = url.pathname;

  // Public paths — no auth required
  if (
    path === '/login' ||
    path === '/login.html' ||
    path.startsWith('/api/')
  ) {
    return;
  }

  const cookieHeader = req.headers.get('cookie') || '';
  const token = parseCookie(cookieHeader, 'fiq_auth');
  const expected = process.env.AUTH_TOKEN;

  if (expected && token === expected) {
    return; // authenticated — pass through
  }

  const loginUrl = new URL('/login', req.url);
  loginUrl.searchParams.set('next', path);
  return Response.redirect(loginUrl.toString(), 302);
}

function parseCookie(header, name) {
  const prefix = name + '=';
  for (const part of header.split(';')) {
    const trimmed = part.trim();
    if (trimmed.startsWith(prefix)) {
      return decodeURIComponent(trimmed.slice(prefix.length));
    }
  }
  return null;
}

export const config = {
  matcher: ['/((?!_vercel|favicon\\.ico).*)'],
};
