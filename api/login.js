export default function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).end();
    return;
  }

  const { password } = req.body || {};
  const SITE_PASSWORD = process.env.SITE_PASSWORD;
  const AUTH_TOKEN = process.env.AUTH_TOKEN;

  if (!SITE_PASSWORD || !AUTH_TOKEN) {
    res.status(503).json({ error: 'Auth not configured' });
    return;
  }

  if (!password || password !== SITE_PASSWORD) {
    // Brief delay to blunt brute-force
    setTimeout(() => res.status(401).json({ error: 'Incorrect password' }), 800);
    return;
  }

  // 30-day session cookie — HttpOnly so JS can't read it (XSS-safe)
  res.setHeader('Set-Cookie',
    `fiq_auth=${AUTH_TOKEN}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=2592000`
  );
  res.status(200).json({ ok: true });
}
