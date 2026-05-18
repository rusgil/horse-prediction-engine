const { readFileSync } = require('fs');
const path = require('path');

const AUTH_TOKEN = process.env.AUTH_TOKEN;

const PAGES = {
  '/':           'pages/index.html',
  '/index.html': 'pages/index.html',
  '/edge':       'pages/edge.html',
  '/edge.html':  'pages/edge.html',
};

module.exports = function handler(req, res) {
  const pathname = (req.url || '/').split('?')[0].split('#')[0];

  const cookie = req.headers.cookie || '';
  const token = parseCookie(cookie, 'fiq_auth');

  if (!AUTH_TOKEN || token !== AUTH_TOKEN) {
    const next = encodeURIComponent(pathname);
    return res.redirect(302, '/login?next=' + next);
  }

  const filePath = PAGES[pathname];
  if (!filePath) {
    return res.status(404).send('Not found');
  }

  try {
    const html = readFileSync(path.join(process.cwd(), filePath), 'utf8');
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.setHeader('Cache-Control', 'private, no-store');
    return res.status(200).send(html);
  } catch (_) {
    return res.status(500).send('Error loading page');
  }
};

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
