/* FunkyIQ shared light/dark theme (2026-07-28).
 *
 * One module, included in the <head> of every page:
 *   <script src="/theme.js"></script>
 * - Applies the saved theme BEFORE first paint (no flash).
 * - Injects light-mode overrides for the union of every page's CSS tokens
 *   (pages stay dark by default; light flips the variables).
 * - Injects ONE Dark/Light toggle into the shared top banner (.page-nav) so
 *   no page carries its own separate control. Preference is shared site-wide
 *   via localStorage key "fiqTheme".
 */
(function () {
  var KEY = 'fiqTheme';
  var saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) {}
  // migrate the old lounge-lab-only key
  if (!saved) {
    try {
      saved = localStorage.getItem('labTheme');
      if (saved) localStorage.setItem(KEY, saved);
    } catch (e) {}
  }
  // Default flipped to LIGHT (2026-07-29): fresh visitors get light; an
  // explicit Dark/Light choice (stored key) always wins.
  var theme = saved === 'dark' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', theme);

  var css = [
    ':root[data-theme="light"] {',
    '  color-scheme: light;',
    '  --bg-base: #eef1f5;',
    '  --bg: #eef1f5;',
    '  --bg-card: #ffffff;',
    '  --surface: #ffffff;',
    '  --surface-2: #e7ebf1;',
    '  --bg-elevated: #e7ebf1;',
    '  --bg-hover: #dde3ea;',
    '  --border: #c6cfda;',
    '  --line: #c6cfda;',
    '  --text: #17212f;',
    '  --text-primary: #17212f;',
    '  --text-secondary: #45536a;',
    '  --text-dim: #5d6b80;',
    '  --text-muted: #8792a3;',
    '  --text-2: #45536a;',
    '  --text-3: #8792a3;',
    '  --gold: #9c7500;',
    '  --gold-light: #8a6a00;',
    '  --gold-mid: rgba(156,117,0,0.35);',
    '  --gold-dim: rgba(156,117,0,0.10);',
    '  --green: #0d9450;',
    '  --green-bright: #0a7a41;',
    '  --green-mid: rgba(13,148,80,0.32);',
    '  --green-dim: rgba(13,148,80,0.10);',
    '  --mint: #0d9450;',
    '  --mint-ink: #ffffff;',
    '  --mint-dim: rgba(13,148,80,0.10);',
    '  --purple: #6d3fd4;',
    '  --purple-bright: #5b2fc0;',
    '  --purple-mid: rgba(109,63,212,0.32);',
    '  --purple-dim: rgba(109,63,212,0.10);',
    '  --red: #cc4444;',
    '  --red-dim: rgba(204,68,68,0.10);',
    '  --blue: #245fc4;',
    '  --blue-dim: rgba(36,95,196,0.10);',
    '  --teal: #0d7a72;',
    '  --teal-bright: #0a938a;',
    '  --amber: #a86400;',
    '  --amber-dim: rgba(168,100,0,0.11);',
    '  --hot: #d05a0c;',
    '  --hot-dim: rgba(208,90,12,0.12);',
    '  --strong: #b02a77;',
    '  --strong-dim: rgba(176,42,119,0.10);',
    '  --shadow: 0 6px 18px rgba(23,33,47,0.08);',
    '}',
    /* Modern brand aurora backdrop layered over each page\'s flat --bg.
       `html body` outranks the pages\' inline `body { background: var(--bg) }`
       regardless of style order, so the tint always shows; the solid --bg
       colour underneath is kept (we only set background-image). */
    /* Fixed pseudo-element instead of background-attachment:fixed, which iOS
       Safari renders once then drops (the "appears then disappears" bug).
       z-index:-1 + pointer-events:none keeps it a non-interactive backdrop. */
    'body::before { content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none; background-image: radial-gradient(1200px 640px at 6% -14%, rgba(124,92,255,0.30), transparent 56%), radial-gradient(1080px 580px at 104% -8%, rgba(56,132,255,0.22), transparent 52%), radial-gradient(1220px 820px at 92% 116%, rgba(236,72,153,0.20), transparent 58%); }',
    ':root[data-theme="light"] body::before { background-image: radial-gradient(1200px 640px at 6% -14%, rgba(124,92,255,0.22), transparent 56%), radial-gradient(1080px 580px at 104% -8%, rgba(56,132,255,0.17), transparent 52%), radial-gradient(1220px 820px at 92% 116%, rgba(236,72,153,0.18), transparent 58%); }',
    /* top banner (.page-nav / .brand-bar use hardcoded dark colours) */
    /* Frosted, translucent menu bar so the aurora backdrop tints through it
       (was a near-opaque flat band). backdrop-blur keeps it readable. */
    'body .page-nav { background: rgba(14,18,32,0.42); backdrop-filter: blur(11px); -webkit-backdrop-filter: blur(11px); }',
    ':root[data-theme="light"] body .page-nav { background: rgba(255,255,255,0.5); backdrop-filter: blur(11px); -webkit-backdrop-filter: blur(11px); border-bottom: 1px solid rgba(213,220,229,0.55); }',
    ':root[data-theme="light"] .page-nav-btn { color: #5d6b80; background: rgba(23,33,47,0.04); border-color: rgba(23,33,47,0.12); }',
    ':root[data-theme="light"] .page-nav-btn:hover { background: rgba(23,33,47,0.08); color: #17212f; }',
    ':root[data-theme="light"] .page-nav-btn.edge.active   { background: rgba(13,148,80,0.12);  color: #0a7a41; border-color: rgba(13,148,80,0.4); }',
    ':root[data-theme="light"] .page-nav-btn.hot.active    { background: rgba(208,90,12,0.12);  color: #b34d0a; border-color: rgba(208,90,12,0.4); }',
    ':root[data-theme="light"] .page-nav-btn.funk.active   { background: rgba(109,63,212,0.12); color: #5b2fc0; border-color: rgba(109,63,212,0.4); }',
    ':root[data-theme="light"] .page-nav-btn.labs.active   { background: rgba(13,122,114,0.12); color: #0d7a72; border-color: rgba(13,122,114,0.4); }',
    ':root[data-theme="light"] .page-nav-btn.lounge.active { background: rgba(156,117,0,0.14);  color: #8a6a00; border-color: rgba(156,117,0,0.4); }',
    ':root[data-theme="light"] .page-nav-btn.about, :root[data-theme="light"] .page-nav-btn.admin { color: #8792a3; }',
    ':root[data-theme="light"] .page-nav-btn.about:hover, :root[data-theme="light"] .page-nav-btn.admin:hover { color: #45536a; }',
    ':root[data-theme="light"] .brand-bar-logo { color: #c99a0c; }',
    ':root[data-theme="light"] .brand-bar-dot { background: #9c7500; }',
    ':root[data-theme="light"] .brand-bar-subtitle { color: #5d6b80; }',
    /* light mode: card borders read too thin on white — thicken + darken.
       Outcome-coloured borders (res-win/place/miss) keep their colours:
       we only darken the DEFAULT border and bump widths. */
    ':root[data-theme="light"] .pick-card, :root[data-theme="light"] .runner-card, :root[data-theme="light"] .jcard, :root[data-theme="light"] .mc, :root[data-theme="light"] .hero { border-width: 1.5px; }',
    ':root[data-theme="light"] .pick-card:not([class*="res-"]) { border-color: #b7c2cf; }',
    ':root[data-theme="light"] .jcard:not([class*="res-"]) { border-color: #b7c2cf; }',
    ':root[data-theme="light"] .runner-card { border-color: #b7c2cf; }',
    ':root[data-theme="light"] .pick { border-width: 1.5px; border-color: #b7c2cf; }',
    ':root[data-theme="light"] .play:not(.hero) { border-width: 1.5px; border-color: #b7c2cf; }',
    ':root[data-theme="light"] .race, :root[data-theme="light"] .summary-card, :root[data-theme="light"] .strategy-block-box { border-width: 1.5px; border-color: #b7c2cf; }',
    /* brand strip: left-justified (body-prefixed to outrank each page's own
       centre rule) and one typeface everywhere — pages' body fonts differ
       (Outfit vs Barlow), which made the FunkyIQ title shift between pages */
    'body .brand-bar { justify-content: flex-start; }',
    'body .brand-bar-logo, body .brand-bar-subtitle { font-family: \'Outfit\', \'Barlow\', system-ui, sans-serif; }',
    /* "Powered by FunkyIQ™" pill — matches the mynrl.tips treatment
       (grey label + gold FUNKYIQ in a rounded translucent pill). */
    'body .brand-bar-powered {',
    '  display: inline-flex; align-items: center; gap: 5px; text-decoration: none;',
    '  font-family: \'Barlow Condensed\', \'Outfit\', system-ui, sans-serif; font-weight: 700;',
    '  font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; line-height: 1;',
    '  color: #8892a4; background: rgba(255,255,255,0.045);',
    '  border: 1px solid rgba(255,255,255,0.14); padding: 5px 11px; border-radius: 999px;',
    '  white-space: nowrap; transition: border-color 0.15s ease, background 0.15s ease; }',
    'body .brand-bar-powered b { color: #f5c842; font-weight: 800; }',
    'body .brand-bar-powered:hover { border-color: rgba(245,200,66,0.4); background: rgba(245,200,66,0.08); }',
    ':root[data-theme="light"] body .brand-bar-powered { color: #5d6b80; background: rgba(23,33,47,0.04); border-color: rgba(23,33,47,0.15); }',
    ':root[data-theme="light"] body .brand-bar-powered b { color: #c99a0c; }',
    /* Mobile: stack the brand — "Horse Racing Intelligence" over the pill;
       and drop the Login / Join button onto its own line UNDER the toggle. */
    '@media (max-width: 560px) {',
    '  body .brand-bar { flex-direction: column; align-items: flex-start; gap: 6px; }',
    '  .fiq-tr-top { flex-direction: column; align-items: stretch; row-gap: 6px; }',
    '  .fiq-tr-top .fiq-acct { order: 0; align-self: flex-end; }',
    '  .fiq-tr-top .fiq-theme-toggle { order: 1; }',
    '  .fiq-tr-top .fiq-login { order: 2; justify-content: center; }',
    '}',
    /* top-right cluster: About link sits to the LEFT of the theme toggle */
    '.fiq-topright {',
    '  position: absolute; top: 10px; right: 12px; z-index: 120;',
    '  display: inline-flex; flex-direction: column; align-items: flex-end; gap: 6px;',
    '}',
    /* Account group + toggle sit in a row; toggle is pushed to the far right. */
    '.fiq-tr-top { display: inline-flex; align-items: center; gap: 10px; }',
    /* "Hello <name>" + circular person avatar (replaces the nav Account link). */
    '.fiq-acct { display: none; align-items: center; gap: 8px; text-decoration: none; cursor: pointer; }',
    '.fiq-acct.on { display: inline-flex; }',
    '.fiq-hello { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.01em; white-space: nowrap;',
    '  color: var(--text-primary, var(--text, #e8ecf4)); }',
    '.fiq-avatar {',
    '  width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;',
    '  display: inline-flex; align-items: center; justify-content: center;',
    '  background: var(--bg-elevated, var(--surface-2, #1a2130));',
    '  border: 1px solid var(--border, var(--line, #232c3d));',
    '  color: var(--text-primary, var(--text, #e8ecf4));',
    '  transition: border-color 0.15s, box-shadow 0.15s;',
    '}',
    '.fiq-acct:hover .fiq-avatar { border-color: var(--green, #22c55e); box-shadow: 0 0 0 3px rgba(34,197,94,0.14); }',
    /* Login / Join button — shown to logged-out visitors, left of the toggle. */
    '.fiq-login { display: none; text-decoration: none; white-space: nowrap; cursor: pointer;',
    '  font-family: \'Barlow Condensed\', system-ui, sans-serif; font-weight: 600; font-size: 0.72rem;',
    '  letter-spacing: 0.04em; text-transform: uppercase; padding: 4px 12px; border-radius: 999px;',
    '  color: #17a34a; background: transparent;',
    '  border: 1px solid rgba(23,163,74,0.55);',
    '  transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease; }',
    '.fiq-login.on { display: inline-flex; align-items: center; gap: 5px; }',
    '.fiq-login:hover { background: rgba(23,163,74,0.10); border-color: rgba(23,163,74,0.85); }',
    ':root[data-theme=\'dark\'] .fiq-login { color: #4ade80; border-color: rgba(74,222,128,0.45); }',
    ':root[data-theme=\'dark\'] .fiq-login:hover { background: rgba(74,222,128,0.12); border-color: rgba(74,222,128,0.75); }',
    /* The old nav-row Account text button is replaced by the avatar. */
    '.page-nav a[href="/account"] { display: none !important; }',
    '.fiq-topright-link {',
    '  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.02em;',
    '  color: var(--text-dim, var(--text-3, #6b7688)); text-decoration: none;',
    '  white-space: nowrap;',
    '}',
    '.fiq-topright-link:hover { color: var(--text-primary, var(--text, #e8ecf4)); }',
    /* Desktop: Sign in/out as a link at the right end of the menu bar. */
    '.fiq-nav-auth {',
    '  white-space: nowrap; text-decoration: none;',
    '  font-size: 0.8rem; font-weight: 700; letter-spacing: 0.02em;',
    '  color: var(--text-dim, var(--text-3, #6b7688));',
    '}',
    '.fiq-nav-auth:hover { color: var(--text-primary, var(--text, #e8ecf4)); }',
    '@media (max-width: 760px) { .fiq-auth-desktop { display: none !important; } }',
    // Desktop: pin Sign in/out to the right of the menu bar, vertically centred,
    // WITHOUT shifting the centred nav buttons (absolute, not a flex item).
    '@media (min-width: 761px) {',
    '  .fiq-auth-mobile { display: none !important; }',
    '  .page-nav { position: relative; }',
    '  .fiq-nav-auth { position: absolute; right: 14px; top: 50%; transform: translateY(-50%); }',
    '}',
    /* Top-left brand: drop the leading dot. */
    '.brand-bar-dot { display: none !important; }',
    /* toggle control — works on both themes */
    '.fiq-theme-toggle {',
    '  display: inline-flex; align-items: center; gap: 2px;',
    '  background: var(--bg-elevated, var(--surface-2, #1a2130));',
    '  border: 1px solid var(--border, var(--line, #232c3d));',
    '  border-radius: 999px; padding: 2px; vertical-align: middle; flex-shrink: 0;',
    '}',
    '.fiq-theme-toggle button {',
    '  border: none; background: none; cursor: pointer; font: inherit;',
    '  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.02em;',
    '  color: var(--text-dim, var(--text-3, #6b7688));',
    '  border-radius: 999px; padding: 3px 10px; line-height: 1.2;',
    '}',
    '.fiq-theme-toggle button.on {',
    '  background: var(--bg-card, var(--surface, #121722));',
    '  color: var(--text-primary, var(--text, #e8ecf4));',
    '  box-shadow: 0 1px 3px rgba(0,0,0,0.25);',
    '}',
  ].join('\n');
  var styleEl = document.createElement('style');
  styleEl.id = 'fiq-theme-css';
  styleEl.textContent = css;
  document.head ? document.head.appendChild(styleEl)
                : document.documentElement.appendChild(styleEl);

  function apply(t) {
    theme = t === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem(KEY, theme); } catch (e) {}
    var w = document.querySelector('.fiq-theme-toggle');
    if (w) {
      var btns = w.querySelectorAll('button');
      for (var i = 0; i < btns.length; i++) {
        btns[i].classList.toggle('on', btns[i].getAttribute('data-t') === theme);
      }
    }
  }

  function mount() {
    if (document.querySelector('.fiq-topright')) return;
    var host = document.body;
    if (!host) return;

    var bar = document.createElement('span');
    bar.className = 'fiq-topright';

    // Top row: "Hello <name>" + account avatar on the left, Dark/Light toggle
    // pushed to the far right.
    var topRow = document.createElement('span');
    topRow.className = 'fiq-tr-top';

    // Account: greeting + circular person avatar — revealed for signed-in
    // members by gateAuthUi().
    var acct = document.createElement('a');
    acct.className = 'fiq-acct';
    acct.href = '/account';
    acct.setAttribute('aria-label', 'Account');
    acct.setAttribute('title', 'Account');
    acct.innerHTML =
      '<span class="fiq-hello"></span>' +
      '<span class="fiq-avatar"><svg viewBox="0 0 24 24" width="17" height="17" fill="currentColor" aria-hidden="true">' +
      '<path d="M12 12a5 5 0 1 0-5-5 5 5 0 0 0 5 5zm0 2c-4.4 0-8 2.2-8 5v1h16v-1c0-2.8-3.6-5-8-5z"/></svg></span>';
    topRow.appendChild(acct);

    // Login / Join — shown to logged-out visitors (left of the toggle).
    var login = document.createElement('a');
    login.className = 'fiq-login';
    login.href = '/login';
    login.textContent = 'Login / Join';
    topRow.appendChild(login);

    var wrap = document.createElement('span');
    wrap.className = 'fiq-theme-toggle';
    wrap.setAttribute('role', 'radiogroup');
    wrap.setAttribute('aria-label', 'Colour theme');
    ['dark', 'light'].forEach(function (t) {
      var b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('data-t', t);
      b.setAttribute('role', 'radio');
      b.textContent = t === 'dark' ? 'Dark' : 'Light';
      if (t === theme) b.classList.add('on');
      b.addEventListener('click', function () { apply(t); });
      wrap.appendChild(b);
    });
    topRow.appendChild(wrap);

    bar.appendChild(topRow);
    host.appendChild(bar);
  }

  /* One auth check drives the top-right UI:
   *  - Account nav link: hidden for anon, shown once signed in.
   *  - Auth link: "Sign in" (anon) → "Sign out" (signed in, POSTs logout). */
  function gateAuthUi() {
    var circle = document.querySelector('.fiq-acct');
    var login = document.querySelector('.fiq-login');
    fetch('/api/auth/me', { credentials: 'include' })
      .then(function (r) {
        if (!r.ok) return null;
        return r.json().catch(function () { return {}; });
      })
      .then(function (u) {
        var signedIn = !!u;
        if (circle) {
          if (signedIn) {
            var hello = circle.querySelector('.fiq-hello');
            var fn = (u.first_name || '').trim();
            if (hello) { hello.textContent = fn ? ('Hello ' + fn) : ''; hello.style.display = fn ? '' : 'none'; }
            circle.classList.add('on');
          } else {
            circle.classList.remove('on');
          }
        }
        // Login/Join only for logged-out visitors. Sign out lives on /account.
        if (login) login.classList.toggle('on', !signedIn);
      })
      .catch(function () { /* leave default (login hidden) */ });
  }

  /* Admin-only nav: Labs lives under the admin section (2026-07-30). Its
   * nav link is hidden for everyone and only revealed once an admin secret
   * is present in this session (set by the /dashboard or Labs login gate,
   * shared via sessionStorage key "fiq_admin_secret"). Any element tagged
   * [data-admin-only] follows the same rule. */
  function gateAdminNav() {
    var isAdmin = false;
    try { isAdmin = !!sessionStorage.getItem('fiq_admin_secret'); } catch (e) {}
    if (isAdmin) return;
    var sel = '.page-nav-btn.labs, [data-admin-only]';
    var nodes = document.querySelectorAll(sel);
    for (var i = 0; i < nodes.length; i++) nodes[i].style.display = 'none';
  }

  function init() { mount(); gateAdminNav(); gateAuthUi(); }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  /* follow changes made in another tab */
  window.addEventListener('storage', function (e) {
    if (e.key === KEY && e.newValue && e.newValue !== theme) apply(e.newValue);
  });
})();
