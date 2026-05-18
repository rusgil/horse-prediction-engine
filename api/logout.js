export default function handler(req, res) {
  res.setHeader('Set-Cookie',
    'fiq_auth=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT; HttpOnly; Secure; SameSite=Strict'
  );
  res.redirect(302, '/login');
}
