// Site-wide warning banner injector. Every user-facing page includes
// this script; it fetches /api/config/site-warnings and, when a warning
// is active, prepends a fixed-height banner above the page body. Toggle
// controlled via the admin dashboard (POST /api/admin/site-warnings/*).
(function() {
  var API = window.SITE_WARNINGS_API || '';
  function render(w) {
    if (!w || !w.model_unstable || !w.model_unstable.active) return;
    if (document.getElementById('sitePredictionsSplash')) return;  // idempotent
    // Generic splash only — never expose the internal reason (breaker_open,
    // distribution:*) to users. Backend sets `message` to the generic copy.
    var msg = w.model_unstable.message || 'Predictions temporarily unavailable';
    var splash = document.createElement('div');
    splash.id = 'sitePredictionsSplash';
    splash.setAttribute('role', 'status');
    splash.style.cssText = [
      'position:fixed', 'inset:0', 'z-index:9998',
      'display:flex', 'align-items:center', 'justify-content:center',
      'padding:24px', 'text-align:center',
      'background:rgba(7,9,15,0.94)',
      'backdrop-filter:blur(6px)', '-webkit-backdrop-filter:blur(6px)'
    ].join(';');
    splash.innerHTML =
      '<div style="max-width:420px;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;color:#e5e7eb">' +
        '<div style="font-size:44px;line-height:1;margin-bottom:18px">⏳</div>' +
        '<div style="font-size:20px;font-weight:700;letter-spacing:0.01em;margin-bottom:10px">' + msg + '</div>' +
        '<div style="font-size:14px;line-height:1.6;color:#9ca3af">We\'re refreshing the model. Your picks will be back shortly — please check again in a few minutes.</div>' +
      '</div>';
    if (document.body) document.body.appendChild(splash);
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
  // WARNING-ONLY chips: Good/Firm meetings render nothing — a green chip
  // on every card clashed with the green result markers and was noise on
  // all-Good days. No chip = track is Good/Firm.
  var COLORS = { soft: '#e8a020', heavy: '#ef4444' };
  var LABELS = { soft: 'Soft track', heavy: 'Heavy track' };
  return {
    ready: ready,
    // meeting record for a venue name (fuzzy: case/punctuation-insensitive)
    lookup: function(venue) { return (map && map[norm(venue)]) || null; },
    // inline chip: '● Soft track' / '● Heavy track'; '' for good/unknown
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
