module.exports = function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).end();
  }

  const { password } = req.body || {};
  const SITE_PASSWORD = process.env.SITE_PASSWORD;
  const AUTH_TOKEN = process.env.AUTH_TOKEN;

  if (!SITE_PASSWORD || !AUTH_TOKEN) {
    return res.status(503).json({ error: 'Auth not configured' });
  }

  if (!password || password !== SITE_PASSWORD) {
    return setTimeout(() => res.status(401).json({ error: 'Incorrect password' }), 800);
  }

  res.setHeader('Set-Cookie',
    'fiq_auth=' + AUTH_TOKEN + '; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=2592000'
  );
  return res.status(200).json({ ok: true });
};
