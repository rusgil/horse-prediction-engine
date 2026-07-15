// Site-wide warning banner injector. Every user-facing page includes
// this script; it fetches /api/config/site-warnings and, when a warning
// is active, prepends a fixed-height banner above the page body. Toggle
// controlled via the admin dashboard (POST /api/admin/site-warnings/*).
(function() {
  var API = window.SITE_WARNINGS_API || 'https://web-production-dec62.up.railway.app';
  function render(w) {
    if (!w || !w.model_unstable || !w.model_unstable.active) return;
    var msg = w.model_unstable.message || 'Model not stable today — NO BETS';
    var banner = document.createElement('div');
    banner.id = 'siteModelUnstableBanner';
    banner.style.cssText = [
      'background:#dc2626',
      'color:#fff',
      'font-weight:700',
      'text-align:center',
      'padding:10px 16px',
      'font-size:14px',
      'letter-spacing:0.02em',
      'border-bottom:2px solid #7f1d1d',
      'position:sticky',
      'top:0',
      'z-index:9999',
      'box-shadow:0 2px 6px rgba(0,0,0,0.35)'
    ].join(';');
    banner.textContent = '⚠ ' + msg;
    if (document.body) document.body.insertBefore(banner, document.body.firstChild);
  }
  function init() {
    fetch(API + '/api/config/site-warnings')
      .then(function(r) { return r.ok ? r.json() : null; })
      .then(function(w) { if (w) render(w); })
      .catch(function() { /* silent — banner is optional */ });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
