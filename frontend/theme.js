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
  var theme = saved === 'light' ? 'light' : 'dark';
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
    '  --border: #d5dce5;',
    '  --line: #d5dce5;',
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
    /* toggle control — works on both themes */
    '.fiq-theme-toggle {',
    '  display: inline-flex; align-items: center; gap: 2px; margin-left: 10px;',
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
    if (document.querySelector('.fiq-theme-toggle')) return;
    var host = document.querySelector('.page-nav') || document.querySelector('.hd-row')
            || document.querySelector('header') || document.body;
    if (!host) return;
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
    host.appendChild(wrap);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }

  /* follow changes made in another tab */
  window.addEventListener('storage', function (e) {
    if (e.key === KEY && e.newValue && e.newValue !== theme) apply(e.newValue);
  });
})();
