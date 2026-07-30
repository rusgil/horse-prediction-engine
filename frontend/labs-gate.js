/* FunkyIQ Labs admin gate (2026-07-30).
 *
 * Labs now lives under the admin section. Include on every Labs page:
 *   <script src="/labs-gate.js"></script>
 * It drops an opaque login overlay over the page until a valid admin secret
 * is supplied. The secret is the CRON_SECRET (same one the /dashboard uses)
 * and is shared via sessionStorage key "fiq_admin_secret", so signing in on
 * the dashboard also unlocks Labs (and vice-versa). Validation is a real
 * server check — a header-gated admin endpoint that 403s on a bad secret —
 * so this is not a client-only string compare.
 *
 * NOTE: this gates the PAGE. The underlying data endpoints (/api/edge, …)
 * stay public by design (see the 2026-07-30 "page gate only" decision).
 */
(function () {
  var KEY = 'fiq_admin_secret';
  var API = '';
  var PROBE = API + '/api/admin/calibration/history?limit=1';

  function el(id) { return document.getElementById(id); }

  function injectCss() {
    var css = [
      '.labs-gate-overlay {',
      '  position: fixed; inset: 0; z-index: 100000;',
      '  display: flex; align-items: flex-start; justify-content: center;',
      '  padding: 84px 18px 18px;',
      '  background: var(--bg-base, var(--bg, #0b0e13));',
      '  font-family: "Outfit", "Barlow", system-ui, sans-serif;',
      '}',
      '.labs-gate-box {',
      '  width: 100%; max-width: 400px;',
      '  background: var(--bg-card, var(--surface, #121722));',
      '  border: 1px solid var(--border, var(--line, #232c3d));',
      '  border-radius: var(--radius, 14px);',
      '  padding: 32px; text-align: center;',
      '  box-shadow: 0 12px 40px rgba(0,0,0,0.35);',
      '}',
      '.labs-gate-box h2 {',
      '  font-family: "Barlow Condensed", "Barlow", sans-serif;',
      '  font-size: 1.4rem; font-weight: 900;',
      '  letter-spacing: 0.06em; text-transform: uppercase;',
      '  margin: 0 0 8px; color: var(--text-primary, var(--text, #e8ecf4));',
      '}',
      '.labs-gate-box p { font-size: 0.8rem; color: var(--text-dim, var(--text-3, #8792a3)); margin: 0 0 20px; }',
      '.labs-gate-box input {',
      '  width: 100%; box-sizing: border-box;',
      '  background: var(--bg-elevated, var(--surface-2, #1a2130));',
      '  border: 1px solid var(--border, var(--line, #232c3d));',
      '  border-radius: 8px; color: var(--text-primary, var(--text, #e8ecf4));',
      '  font: inherit; font-size: 0.9rem; padding: 10px 14px;',
      '  outline: none; margin-bottom: 12px; transition: border-color .15s;',
      '}',
      '.labs-gate-box input:focus { border-color: var(--blue, #4a9eff); }',
      '.labs-gate-box button {',
      '  width: 100%; background: var(--blue, #2563eb); color: #fff;',
      '  border: none; cursor: pointer;',
      '  font-family: "Barlow Condensed", "Barlow", sans-serif;',
      '  font-size: 1rem; font-weight: 800; letter-spacing: 0.1em;',
      '  text-transform: uppercase; padding: 11px; border-radius: 8px;',
      '  transition: opacity .15s;',
      '}',
      '.labs-gate-box button:hover { opacity: 0.88; }',
      '.labs-gate-box button:disabled { opacity: 0.5; cursor: default; }',
      '.labs-gate-err { font-size: 0.75rem; color: #f87171; margin-top: 10px; min-height: 1em; }',
    ].join('\n');
    var s = document.createElement('style');
    s.textContent = css;
    (document.head || document.documentElement).appendChild(s);
  }

  function buildGate() {
    var g = document.createElement('div');
    g.className = 'labs-gate-overlay';
    g.id = 'labsGate';
    g.innerHTML =
      '<div class="labs-gate-box">' +
      '  <h2>Labs · Admin</h2>' +
      '  <p>Enter the admin secret to open Labs.</p>' +
      '  <input type="password" id="labsSecret" placeholder="Admin secret" autocomplete="off" autocapitalize="off" spellcheck="false">' +
      '  <button id="labsSecretBtn" type="button">Unlock</button>' +
      '  <div class="labs-gate-err" id="labsErr"></div>' +
      '</div>';
    document.body.appendChild(g);
    el('labsSecretBtn').addEventListener('click', submit);
    el('labsSecret').addEventListener('keydown', function (e) {
      if (e.key === 'Enter') submit();
    });
    el('labsSecret').focus();
  }

  function reveal() {
    var g = el('labsGate');
    if (g) g.parentNode.removeChild(g);
    // theme.js hides admin-only nav (incl. the Labs link) before unlock —
    // un-hide it now so the active page's own nav is consistent.
    var nodes = document.querySelectorAll('.page-nav-btn.labs, [data-admin-only]');
    for (var i = 0; i < nodes.length; i++) nodes[i].style.display = '';
  }

  function validate(secret) {
    if (!secret) return Promise.resolve(false);
    return fetch(PROBE, { headers: { 'x-cron-secret': secret } })
      .then(function (r) { return r.ok; })   // 401/403 => false
      .catch(function () { return false; });
  }

  function submit() {
    var input = el('labsSecret');
    var btn = el('labsSecretBtn');
    var err = el('labsErr');
    var v = (input.value || '').trim();
    if (!v) return;
    btn.disabled = true; err.textContent = '';
    validate(v).then(function (ok) {
      btn.disabled = false;
      if (ok) {
        try { sessionStorage.setItem(KEY, v); } catch (e) {}
        reveal();
      } else {
        err.textContent = 'Incorrect secret — try again.';
        input.value = ''; input.focus();
      }
    });
  }

  function boot() {
    injectCss();
    buildGate();
    var s = '';
    try { s = sessionStorage.getItem(KEY) || ''; } catch (e) {}
    if (s) {
      validate(s).then(function (ok) {
        if (ok) reveal();
        else { try { sessionStorage.removeItem(KEY); } catch (e) {} }
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
