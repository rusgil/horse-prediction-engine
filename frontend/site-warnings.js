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

// ── Shared track-conditions lookup ─────────────────────────────────────
// Going underpins model confidence, so every surface shows it: the
// Lounge at meeting level, and a chip on each pick card (Edge, Hot Seat,
// Funk Me Up, Labs). Pages await FIQConditions.ready once at the start
// of their render, then call FIQConditions.chip(venue) inline.
// Data: /api/conditions-today — meetings[] with worst-seen category.
window.FIQConditions = (function() {
  var API = window.SITE_WARNINGS_API || '';
  var map = null;
  function norm(v) { return String(v || '').toLowerCase().replace(/[^a-z]/g, ''); }
  var ready = fetch(API + '/api/conditions-today')
    .then(function(r) { return r.ok ? r.json() : null; })
    .then(function(c) {
      map = {};
      ((c && c.meetings) || []).forEach(function(m) { map[norm(m.venue)] = m; });
      return map;
    })
    .catch(function() { map = {}; return map; });
  var COLORS = { good: '#22c55e', soft: '#e8a020', heavy: '#ef4444' };
  var LABELS = { good: 'Good', soft: 'Soft', heavy: 'Heavy' };
  return {
    ready: ready,
    // meeting record for a venue name (fuzzy: case/punctuation-insensitive)
    lookup: function(venue) { return (map && map[norm(venue)]) || null; },
    // small inline chip: '● Good' colored by category; '' when unknown
    chip: function(venue) {
      var m = map && map[norm(venue)];
      if (!m || !COLORS[m.category]) return '';
      var col = COLORS[m.category];
      return '<span class="fiq-going-chip" style="display:inline-flex;align-items:center;gap:4px;padding:1px 7px;border-radius:999px;border:1px solid ' + col + '55;background:' + col + '18;color:' + col + ';font-size:0.68rem;font-weight:600;white-space:nowrap;vertical-align:middle;margin-left:6px">' +
        '<span style="width:6px;height:6px;border-radius:50%;background:' + col + ';display:inline-block"></span>' +
        LABELS[m.category] + '</span>';
    }
  };
})();
