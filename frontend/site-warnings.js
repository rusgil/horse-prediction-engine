// Site-wide warning banner injector. Every user-facing page includes
// this script; it fetches /api/config/site-warnings and, when a warning
// is active, prepends a fixed-height banner above the page body. Toggle
// controlled via the admin dashboard (POST /api/admin/site-warnings/*).
(function() {
  var API = window.SITE_WARNINGS_API || '';
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
      'border-top:2px solid #7f1d1d',
      'border-bottom:2px solid #7f1d1d',
      'z-index:9998',
      'box-shadow:0 2px 6px rgba(0,0,0,0.35)'
    ].join(';');
    banner.textContent = '⚠ ' + msg;
    // Insert immediately AFTER the FunkyIQ brand bar so the top-of-page
    // chrome ("FunkyIQ / Horse Racing") remains visible above the alert.
    // Falls back to first child of body if the brand bar isn't present
    // (defensive — every user page currently ships one).
    var brand = document.querySelector('.brand-bar');
    if (brand && brand.parentNode) {
      if (brand.nextSibling) {
        brand.parentNode.insertBefore(banner, brand.nextSibling);
      } else {
        brand.parentNode.appendChild(banner);
      }
    } else if (document.body) {
      document.body.insertBefore(banner, document.body.firstChild);
    }
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
