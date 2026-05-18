import { readFileSync } from 'fs';
import { join } from 'path';

const AUTH_TOKEN = process.env.AUTH_TOKEN;

const PAGES = {
  '/':           'frontend/index.html',
  '/index.html': 'frontend/index.html',
  '/edge':       'frontend/edge.html',
  '/edge.html':  'frontend/edge.html',
};

export default function handler(req, res) {
  const pathname = (req.url || '/').split('?')[0].split('#')[0];

  // Auth check — read HttpOnly cookie set by /api/login
  const cookie = req.headers.cookie || '';
  const token = parseCookie(cookie, 'fiq_auth');

  if (!AUTH_TOKEN || token !== AUTH_TOKEN) {
    const next = encodeURIComponent(pathname === '/' ? '/' : pathname);
    return res.redirect(302, `/login?next=${next}`);
  }

  const filePath = PAGES[pathname];
  if (!filePath) {
    return res.status(404).send('Not found');
  }

  try {
    const html = readFileSync(join(process.cwd(), filePath), 'utf8');
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.setHeader('Cache-Control', 'private, no-store');
    return res.status(200).send(html);
  } catch (_) {
    return res.status(500).send('Error loading page');
  }
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
