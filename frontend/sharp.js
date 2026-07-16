// Shared Sharp classification helpers.
//
// Single source of truth for "is this pick Sharp?" and "is this race
// Sharp?" — used by Edge, Hot Seat, and the Lounge. Prior to this
// module each page had its own copy of the logic and they drifted
// (Hot Seat bundled a value-bet check into Sharp, Edge didn't;
// Lounge counted only settled Sharp races when classifying meetings).
//
// Server-side truth is `is_sharp: bool` on race objects, pick objects,
// and history rows. The gate the server applies is:
//   (rank-1 win_prob >= 30% OR top-3 sum >= 60%)
//   AND days_since_last_run <= 180
// See _snapshot_prerace_predictions in main.py.
//
// Everything below either delegates to that flag or falls back to the
// same gate when the flag is absent (old cached rows).
(function (global) {
  'use strict';

  // Whether a race object (from /api/meetings/{date}/{venue}) or a pick
  // object (from /api/edge) is Sharp. Same shape both directions.
  function isSharp(obj) {
    if (!obj) return false;
    if (typeof obj.is_sharp === 'boolean') return obj.is_sharp;
    // Fallback: mirror the server gate for older cached rows that
    // predate is_sharp being shipped in the API response.
    const modelPct =
      (obj.model_pct != null ? obj.model_pct
        : obj.top_win_probability != null ? obj.top_win_probability * 100
        : 0);
    const field = obj.field || [];
    const top3Sum = field
      .filter(function (f) { return !f.scratched; })
      .sort(function (a, b) { return (b.win_pct || 0) - (a.win_pct || 0); })
      .slice(0, 3)
      .reduce(function (s, f) { return s + (f.win_pct || 0); }, 0);
    const confident = modelPct >= 30 || top3Sum >= 60;
    const daysOff = obj.days_since_last_run;
    const layoffOk = daysOff == null || daysOff <= 180;
    return confident && layoffOk;
  }

  // Export under a stable namespace so pages can call FIQSharp.isSharp(x)
  // without polluting the global name — one shorter alias too because
  // isSharpPick / isSharpEdgePick / _isSharpRace read easily in the
  // existing filter chains.
  global.FIQSharp = { isSharp: isSharp };
  global.isSharpPick = isSharp;
  global.isSharpEdgePick = isSharp;
})(window);
