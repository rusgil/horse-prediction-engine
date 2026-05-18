module.exports = function handler(req, res) {
  const cookie = req.headers.cookie || '';
  const token = parseCookie(cookie, 'fiq_auth');
  const AUTH_TOKEN = process.env.AUTH_TOKEN;

  if (AUTH_TOKEN && token === AUTH_TOKEN) {
    return res.status(200).json({ ok: true });
  }
  return res.status(401).json({ ok: false });
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
