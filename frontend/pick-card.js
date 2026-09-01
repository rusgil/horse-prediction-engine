/* Shared pick card — single source of truth for the Edge + Hot Seat card.
 *
 * Both pages build a NORMALIZED `pick` object (see the field list below) and call
 * window.PickCard.render(pick, ctx). The card owns its structure, chips and the
 * big centred WIN/PLACE stat so the two pages can never drift again.
 *
 * Normalized pick fields:
 *   race_id, venue, race_number, horse_name, scheduled_time
 *   win_pct, place_pct                     (0-100, or null)
 *   rank2_pct, rank3_pct, rank4_pct, rank5_pct   (0-100, for clear-fav/exotic)
 *   distance, barrier, weight, jockey, trainer, jockey_pct, trainer_pct
 *   form_figures, wins_last_10, places_last_10, starts_last_10, days_off
 *   odds, confidence_tier, going_offtrack, is_sharp, is_premium
 *   place_play  ({paid_place_pct} | null)
 *   sportsbet_available, cancelled, streak ({badge:'hot'|'warm'} | null)
 *   result  ({winner,placed,position,sp,place_odds,scratched,no_result,winner_name} | null)
 *
 * ctx: { venueBadge?: (venue)=>html, triStrip?: (pick)=>html, showSpark?: bool }
 */
(function () {
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const wallTime = iso => {
    if (!iso) return '';
    try { return new Date(iso).toLocaleTimeString('en-AU', { hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'Australia/Sydney' }).replace(' ', '').toLowerCase(); }
    catch (e) { return ''; }
  };
  const countdown = iso => {
    if (!iso) return { label: '', urg: '' };
    const mins = Math.round((new Date(iso).getTime() - Date.now()) / 60000);
    if (mins < 0) return { label: '', urg: '' };
    if (mins < 60) return { label: mins + 'm', urg: mins <= 15 ? 'soon' : 'hour' };
    const h = Math.floor(mins / 60), m = mins % 60;
    return { label: h + 'h' + (m ? ' ' + m + 'm' : ''), urg: h < 2 ? 'hour' : 'far' };
  };

  // Model accuracy at a given confidence level — historical actual win rate for
  // rank-1 picks in that model% band (mirrors the server _CALIBRATED_WIN_RATES).
  // Lets Hot Seat show the 3rd stat too without a per-race calibrated field.
  const _CAL = [[50, 88], [45, 82], [40, 76], [35, 71], [30, 66]];
  function calibratedRate(win_pct) {
    if (win_pct == null) return null;
    for (const [t, r] of _CAL) if (win_pct >= t) return r;
    return 66;
  }

  // 3-stat prediction bar (old-edge style): win prediction / place prediction /
  // model accuracy at this level. accuracy is optional (omit → 2 stats).
  // ── Open-race demotion ──────────────────────────────────────────────────
  // A wide-open field has no strong win pick: the model's top pick sits below
  // this win% AND the race isn't Sharp. In these races the win pick is a
  // coin-flip (historically ~18%) and the PLACE % is high mostly because the
  // field is big — which reads as a "safe" bet and gives false confidence.
  // So we demote the card: no confidence/fav chips, muted stats, honest caveat.
  var OPEN_RACE_WIN_MAX = 25;
  function _pickWinPct(pick) {
    if (pick == null) return null;
    if (pick.win_pct != null) return pick.win_pct;
    if (pick.model_pct != null) return pick.model_pct;
    const p = pick.win_probability != null ? pick.win_probability
      : (pick.top_win_probability != null ? pick.top_win_probability : null);
    if (p == null) return null;
    return p <= 1 ? p * 100 : p;
  }
  // Badged (gold/silver/bronze) venues are an EXCEPTION to the low-confidence
  // rules: the model under-rates them but still wins (<20% picks there hit ~26%,
  // OOS-confirmed), so we neither hide nor demote their races. Loaded once from
  // /api/venue-badges; until it resolves the set is empty (no exceptions yet).
  var _badgedVenues = null, _badgesLoading = null;
  function _normVenue(s) { return (s == null ? '' : String(s)).toLowerCase().replace(/[^a-z0-9]/g, ''); }
  function loadVenueBadges() {
    if (_badgedVenues) return Promise.resolve(_badgedVenues);
    if (_badgesLoading) return _badgesLoading;
    _badgesLoading = fetch('/api/venue-badges').then(r => r.ok ? r.json() : null).then(d => {
      const s = new Set(), b = d && d.badges;
      if (b) for (const k in b) { if (b[k] && b[k].badge && b[k].badge !== 'avoid') s.add(_normVenue(k)); }
      _badgedVenues = s; return s;
    }).catch(() => { _badgedVenues = new Set(); return _badgedVenues; });
    return _badgesLoading;
  }
  function _venueBadged(pick) {
    if (!_badgedVenues || !pick) return false;
    return _badgedVenues.has(_normVenue(pick.venue)) || _badgedVenues.has(_normVenue(pick.venue_code))
      || _badgedVenues.has(_normVenue(pick._venue));
  }
  function isOpenRace(pick) {
    if (!pick || pick.is_sharp || _venueBadged(pick)) return false;
    const w = _pickWinPct(pick);
    // Compare the ROUNDED value the card displays — so a pick shown as "25%"
    // (true 24.5-25.4%) is never treated as below 25 (matches "<25 not <=25").
    return w != null && Math.round(w) < OPEN_RACE_WIN_MAX;
  }
  // Below this the top pick is a coin-flip we don't stand behind — HIDDEN from
  // the pick feeds entirely (matches the overall-stats <20% floor). The 20-25%
  // band still shows, demoted (isOpenRace). Sharp + badged venues never hidden.
  var HIDE_RACE_WIN_MAX = 20;
  function isHiddenRace(pick) {
    if (!pick || pick.is_sharp || _venueBadged(pick)) return false;
    const w = _pickWinPct(pick);
    // Compare the ROUNDED value the card displays — a pick shown as "20%"
    // (true 19.5-20.4%) is NOT below 20 ("<20 not <=20").
    return w != null && Math.round(w) < HIDE_RACE_WIN_MAX;
  }

  function dualStat(win_pct, place_pct, accuracy, open) {
    if (win_pct == null) return '';
    const f = v => (+Number(v).toFixed(1));
    let out = `<div class="ds"><span class="ds-num num ${(!open && win_pct >= 30) ? 'v-win' : ''}">${f(win_pct)}%</span><span class="ds-lbl">win prediction</span></div>`;
    if (place_pct != null) out += `<div class="ds-sep"></div><div class="ds"><span class="ds-num num ${(!open && place_pct >= 50) ? 'v-plc' : ''}">${f(place_pct)}%</span><span class="ds-lbl">place prediction</span></div>`;
    if (accuracy != null) out += `<div class="ds-sep"></div><div class="ds"><span class="ds-num num v-acc">${f(accuracy)}%</span><span class="ds-lbl">model accuracy at this level</span></div>`;
    const note = open ? `<div class="open-note">⚖️ Open race — no strong win pick. A big field lifts the place % (it isn't a strong edge).</div>` : '';
    return `<div class="dual-stat-row${open ? ' open' : ''}">${out}</div>${note}`;
  }

  // Compact 2-up WIN/PLACE for drawer runner rows — side by side, big and bold.
  // Takes fractions (0–1). Sub-threshold numbers keep full --text weight (never
  // dimmed to grey) so they stay legible below 30% win / 50% place.
  function winPlace(winP, placeP) {
    const pc = v => v == null ? null : Math.round(v * 1000) / 10;
    const w = pc(winP), p = pc(placeP);
    if (w == null) return `<div style="font-size:0.68rem;color:var(--text-3)">No prediction yet</div>`;
    let out = `<div class="ds"><span class="ds-num num ${w >= 30 ? 'v-win' : ''}">${w}%</span><span class="ds-lbl">win</span></div>`;
    if (p != null) out += `<div class="ds-sep"></div><div class="ds"><span class="ds-num num ${p >= 50 ? 'v-plc' : ''}">${p}%</span><span class="ds-lbl">place</span></div>`;
    return `<div class="dual-stat-row wp">${out}</div>`;
  }

  function tierChip(pick) {
    const ct = pick.confidence_tier, conf = pick.win_pct || 0;
    if (ct === 'hot' || (ct == null && conf >= 46 && !pick.going_offtrack)) return '<span class="chip tier-hot2">🔥 HOT PICK</span>';
    if (ct === 'high' || (ct == null && conf >= 36 && !pick.going_offtrack)) return '<span class="chip tier-high2">⚡ HIGH CONFIDENCE</span>';
    if (ct === 'strong' || (ct == null && conf >= 30 && !pick.going_offtrack)) return '<span class="chip tier-strong2">✦ STRONG</span>';
    return '';
  }
  function favChip(pick) {
    if (pick.win_pct == null || pick.rank2_pct == null) return '';
    const gap = pick.win_pct - pick.rank2_pct;
    if (gap >= 5) return '<span class="chip cf">👑 CLEAR FAV</span>';
    if (gap < 2 && pick.win_pct >= 25) return '<span class="chip tu">⚖️ TOSS-UP</span>';
    return '';
  }
  function exoticChips(pick) {
    const G = 4; let out = '';
    const q = (pick.rank2_pct != null && pick.rank3_pct != null) ? pick.rank2_pct - pick.rank3_pct : null;
    const t = (pick.rank3_pct != null && pick.rank4_pct != null) ? pick.rank3_pct - pick.rank4_pct : null;
    const f = (pick.rank4_pct != null && pick.rank5_pct != null) ? pick.rank4_pct - pick.rank5_pct : null;
    if (q != null && q >= G) out += '<span class="chip quin">🥈 CLEAR QUINELLA</span>';
    if (t != null && t >= G) out += '<span class="chip tri">🥉 CLEAR TRIFECTA</span>';
    if (f != null && f >= G) out += '<span class="chip ff">🎖️ CLEAR FIRST-4</span>';
    return out;
  }
  function placeChip(pick) {
    const pp = pick.place_play;
    if (!pp || !pp.paid_place_pct) return '';
    const tip = `Weak late country field — our win pick is a coin-flip here, so it's not a Sharp win bet. But it still PLACES (pays) about ${pp.paid_place_pct}% of the time — consider a place bet instead.`;
    return `<span class="chip place-play" tabindex="0" onclick="event.stopPropagation();this.classList.toggle('tip-open')" data-tip="${esc(tip)}">🅿️ PLACE ~${pp.paid_place_pct}% <span class="chip-i">ⓘ</span></span>`;
  }

  // ── Streaming-odds sparkline (shared by every surface). render() emits an
  // empty <div class="spark-slot" id="spark-<race>"> when ctx.showSpark is true;
  // a surface then calls PickCard.fetchSparklines(picks) after rendering to fill
  // them. `picks` items just need race_id + a horse (horse_name or top_pick).
  const _sparkCache = {};   // race_id||horse -> snapshots
  function buildSparklineSvg(snaps) {
    if (!snaps || snaps.length < 2) return '';
    const odds = snaps.map(s => s.win_odds).filter(Boolean);
    if (odds.length < 2) return '';
    const W = 84, H = 24, PAD = 3;
    const minO = Math.min(...odds), maxO = Math.max(...odds), range = (maxO - minO) || 0.01;
    const xs = odds.map((_, i) => PAD + (i / (odds.length - 1)) * (W - 2 * PAD));
    const ys = odds.map(o => PAD + ((maxO - o) / range) * (H - 2 * PAD));
    const pts = xs.map((x, i) => `${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ');
    const first = odds[0], last = odds[odds.length - 1], diff = last - first;
    const col = diff < -0.1 ? 'var(--mint)' : diff > 0.1 ? 'var(--red)' : 'var(--text-3)';
    const lbl = diff < -0.1 ? '▼ firming' : diff > 0.1 ? '▲ drifting' : '— stable';
    return `<div class="odds-spark" title="Odds: opened $${first.toFixed(2)} → now $${last.toFixed(2)}">
      <svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="${xs[xs.length-1].toFixed(1)}" cy="${ys[ys.length-1].toFixed(1)}" r="2.4" fill="${col}"/></svg>
      <span class="spark-lbl" style="color:${col}">${lbl}</span></div>`;
  }
  const _sparkHorse = p => p.horse_name || p.top_pick || null;
  async function fetchSparklines(picks) {
    const up = (picks || []).filter(p => {
      const j = p.scheduled_time ? new Date(p.scheduled_time).getTime() : null;
      return (!j || j > Date.now()) && p.race_id && _sparkHorse(p);
    });
    await Promise.all(up.map(async p => {
      const key = `${p.race_id}||${_sparkHorse(p)}`;
      if (_sparkCache[key]) return;
      try {
        const r = await fetch(`/api/races/${encodeURIComponent(p.race_id)}/odds-trend?horse=${encodeURIComponent(_sparkHorse(p))}`);
        if (r.ok) _sparkCache[key] = (await r.json()).snapshots || [];
      } catch (e) {}
    }));
    up.forEach(p => {
      const snaps = _sparkCache[`${p.race_id}||${_sparkHorse(p)}`];
      if (!snaps || snaps.length < 2) return;
      const slot = document.getElementById('spark-' + p.race_id.replace(/[^a-z0-9]/gi, '-'));
      if (slot) slot.innerHTML = buildSparklineSvg(snaps);
    });
  }

  function render(pick, ctx) {
    ctx = ctx || {};
    const now = Date.now();
    const jump = pick.scheduled_time ? new Date(pick.scheduled_time).getTime() : null;
    const isPast = jump && jump < now;
    const cd = countdown(pick.scheduled_time);
    const res = pick.result || null;
    const settledRes = isPast && res && !res.scratched && !res.no_result;
    let displayOdds, oddsLbl;
    if (settledRes && res.winner) { displayOdds = res.sp; oddsLbl = 'SP'; }
    else if (settledRes && res.placed) { displayOdds = res.place_odds || null; oddsLbl = 'place'; }
    else if (settledRes) { displayOdds = null; oddsLbl = ''; }
    else { displayOdds = pick.odds || (res && res.sp); oddsLbl = 'best odds'; }
    const oddsCls = displayOdds >= 3 ? 'great' : displayOdds >= 2.5 ? 'good' : '';
    const hasResult = res && !res.scratched && !res.no_result && (res.winner === true || res.placed === true || res.position != null);
    const outcome = hasResult ? (res.winner ? 'res-win' : res.placed ? 'res-place' : 'res-miss') : '';
    const dim = (pick.cancelled || (res && res.scratched)) ? 'opacity:0.55;filter:grayscale(0.4);' : '';
    const posLbl = res && res.position ? (res.position === 2 ? '2nd' : res.position === 3 ? '3rd' : res.position < 90 ? res.position + 'th' : '') : '';
    const streak = (() => {
      const s = pick.streak;
      if (!s || isPast) return '';
      if (s.badge === 'hot') return '<span class="streak hotk" title="Won their most recent race — +18pp historical win-rate lift">🔥 WON LAST</span>';
      if (s.badge === 'warm') return '<span class="streak warm" title="Placed top-3 last start — +11pp lift">🌤 PLACED LAST</span>';
      return '';
    })();
    let verdict = '';
    if (pick.cancelled) verdict = '<div class="verdict-line"><span class="vn">⚠️ Race abandoned</span></div>';
    else if (res && res.scratched) verdict = '<div class="verdict-line"><span class="vn">Scratched — no bet</span></div>';
    else if (isPast && hasResult) {
      const winner = (!res.winner && res.winner_name) ? `<span class="winner">Winner: <b>${esc(res.winner_name)}</b></span>` : '';
      if (res.winner) verdict = `<div class="verdict-line"><span class="vw">✓ Won${res.sp ? ' at $' + res.sp.toFixed(2) : ''}</span></div>`;
      else if (res.placed) verdict = `<div class="verdict-line"><span class="vp">~ Placed${posLbl ? ' ' + posLbl : ''}</span>${winner}</div>`;
      else verdict = `<div class="verdict-line"><span class="vl">✗ Did not place${posLbl ? ' (' + posLbl + ')' : ''}</span>${winner}</div>`;
    } else if (isPast) verdict = '<div class="verdict-line"><span class="vn">Result pending…</span></div>';
    const l10 = pick.form_figures
      ? ` · <b class="ff">form ${esc(pick.form_figures)}</b>`
      : ((pick.wins_last_10 != null && pick.places_last_10 != null)
        ? ` · last ${pick.starts_last_10 && pick.starts_last_10 < 10 ? pick.starts_last_10 : 10}: ${pick.wins_last_10}W/${pick.places_last_10} top-3` : '');
    const jt = `${pick.jockey ? 'J: ' + esc(pick.jockey) : ''}${pick.jockey_pct != null ? ` ${Math.round(pick.jockey_pct)}%` : ''}${pick.trainer ? (pick.jockey ? ' · ' : '') + 'T: ' + esc(pick.trainer) : ''}${pick.trainer_pct != null ? ` ${Math.round(pick.trainer_pct)}%` : ''}`;
    const badge = ctx.venueBadge ? (ctx.venueBadge(pick.venue) || '') : '';
    const tri = ctx.triStrip ? (ctx.triStrip(pick) || '') : '';
    const resultMark = (isPast && hasResult)
      ? (res.winner ? '<span class="res win">✓</span>' : res.placed ? '<span class="res place">✓' + (posLbl ? ' ' + posLbl : '') + '</span>' : '<span class="res miss">✗</span>')
      : '';
    const timeTop = isPast
      ? `<div class="when-top num">${wallTime(pick.scheduled_time)}</div>`
      : `<div class="when-top">${cd.label ? `<span class="cd ${cd.urg || ''}">${cd.label}</span> ` : ''}<span class="num">${wallTime(pick.scheduled_time)}</span></div>`;
    const openRace = !isPast && isOpenRace(pick);
    return `<div class="jcard pcard ${outcome} ${openRace ? 'open-race' : ''} ${!isPast && !openRace && pick.confidence_tier === 'hot' ? 'hot' : ''} ${!isPast && !openRace && pick.is_premium ? 'prem' : ''}" style="${dim}" data-open="${esc(pick.race_id)}" data-horse="${esc(pick.horse_name)}">
      <div class="mid">
        ${(() => {
          const idEl = `<div class="pk-venue">${esc(pick.venue)} <b>R${pick.race_number}</b>${badge}</div>`
            + `<div class="pk-horse">${esc(pick.horse_name)}${streak}${resultMark ? ' ' + resultMark : ''}</div>`;
          const metaEl = `<div class="pk-sub num">${[pick.distance ? pick.distance + 'm' : '', pick.barrier ? 'B' + pick.barrier : '', pick.weight ? pick.weight + 'kg' : ''].filter(Boolean).join(' · ')}${l10}</div>`
            + `<div class="pk-sub">${jt}${pick.days_off != null && pick.days_off < 999 ? ` · ${pick.days_off}d off` : ''}</div>`;
          // Name and meta sit side by side on wide screens; they stack under
          // the name on narrow/mobile-portrait via .pk-split's media query.
          return `<div class="pk-split"><div class="pk-id">${idEl}</div><div class="pk-meta">${metaEl}</div></div>`;
        })()}
        <div class="pk-chips">${openRace ? '<span class="chip openrace">⚖️ OPEN RACE</span>' : `${tierChip(pick)}${!isPast && pick.is_premium ? '<span class="chip prem">💎 PREMIUM</span>' : ''}${pick.is_sharp ? '<span class="chip sharp">🎯 SHARP</span>' : ''}${!isPast ? favChip(pick) : ''}${!isPast ? exoticChips(pick) : ''}`}${pick.sportsbet_available === false ? '<span class="chip nosb">🚫 Limited fixed-odds</span>' : ''}</div>
      </div>
      <div class="right">
        ${timeTop}
        ${displayOdds ? `<div class="odds-pill ${oddsCls}"><b class="num">$${(+displayOdds).toFixed(2)}</b>${oddsLbl}</div>` : '<div class="odds-pill">TBA</div>'}
        ${settledRes && res.winner && res.place_odds ? `<div class="odds-pill"><b class="num" style="font-size:1rem;color:var(--blue)">$${(+res.place_odds).toFixed(2)}</b>place</div>` : ''}
        ${!isPast && ctx.showSpark ? `<div class="spark-slot" id="spark-${pick.race_id.replace(/[^a-z0-9]/gi, '-')}"></div>` : ''}
      </div>
      ${dualStat(pick.win_pct, pick.place_pct, pick.accuracy != null ? pick.accuracy : calibratedRate(pick.win_pct), openRace)}
      ${pick.is_teaser ? confidenceCallout(pick) : ''}
      ${!isPast && ctx.footer ? (ctx.footer(pick) || '') : ''}${verdict}${tri}
    </div>`;
  }

  // Finishing-order RESULT block — the exact Lounge drawer format. `fin` is the
  // 1st–4th finishers ([{position,name,sp,place_odds}]). Shared so Edge / Hot
  // Seat / Lounge render identical results.
  function resultBlock(fin, pickName) {
    if (!fin || !fin.length) return '';
    const ORD = { 1: '1st', 2: '2nd', 3: '3rd', 4: '4th' };
    const pn = (pickName || '').toLowerCase();
    return `<div class="dw-result"><div class="dw-result-t">RESULT</div>` + fin.map(f => {
      const isPick = (f.name || '').toLowerCase() === pn;
      return `<div class="dw-res-row ${f.position === 1 ? 'w' : ''} ${isPick ? 'mine' : ''}">
        <span class="pos num">${ORD[f.position] || (f.position + 'th')}</span>
        <span class="nm">${esc(f.name)}${isPick ? ' <span class="pick-tag">OUR PICK</span>' : ''}</span>
        ${f.position === 1
          ? `${f.sp ? `<span class="sp num">$${(+f.sp).toFixed(2)}</span>` : ''}${f.place_odds ? `<span class="sp num" style="color:var(--blue)">$${(+f.place_odds).toFixed(2)}</span>` : ''}`
          : (f.place_odds ? `<span class="sp num" style="color:var(--blue)">$${(+f.place_odds).toFixed(2)}</span>` : '')}
      </div>`;
    }).join('') + `</div>`;
  }

  // ── Freemium paywall (Stage 3, 2026-08-24) ─────────────────────────
  // The API redacts all-but-the-next race to a locked stub (race_id, venue,
  // race_number, scheduled_time, distance, field_size + locked:true) and
  // adds a top-level paywall{active,teaser_race_id}. These helpers render
  // the locked card + banner identically on every page. Billing config
  // comes from /api/config/public — pages call configureBilling(cfg.billing)
  // once. Until Paddle is wired (billing.enabled=false) Unlock → /login.
  let _billing = null;
  function configureBilling(b) { _billing = b || null; loadTierStats(); refreshSplashUpsell(); }

  // Live tier calibration (from /api/track-record) for the free-pick callout —
  // "at this confidence our picks win X% / place Y%". Fetched once per page; the
  // pick's win% maps to a tier band → the tier's REAL historical win/place.
  let _tierStats = null;
  async function loadTierStats() {
    if (_tierStats) return;
    try {
      const r = await fetch('/api/track-record');
      if (r.ok) {
        const d = await r.json();
        _tierStats = (d.tiers || []).map(t => ({
          lo: t.conf_min == null ? 0 : t.conf_min,
          hi: t.conf_max == null ? 999 : t.conf_max,
          win: t.win_pct, place: t.place_pct, n: t.races,
        }));
      }
    } catch (e) { /* callout just won't render */ }
  }
  function _tierFor(winPct) {
    if (!_tierStats || winPct == null) return null;
    return _tierStats.find(t => winPct >= t.lo && winPct <= t.hi) || null;
  }
  function confidenceCallout(pick) {
    const t = _tierFor(pick.win_pct);
    if (!t || !t.n) return '';
    return `<div class="conf-callout">🎯 At this confidence, our picks <b class="cc-win">win ${t.win}%</b> · <b class="cc-plc">place ${t.place}%</b> <span class="cc-n">— ${t.n} picks at this level</span></div>`;
  }
  const _price = () => (_billing && _billing.price != null ? _billing.price : 6.99);
  const _days = () => (_billing && _billing.pass_days != null ? _billing.pass_days : 5);
  // Currency-aware money label, e.g. "US$6.99 (~A$9.90)". The real charge is
  // the USD amount; the (~A$…) note is display-only, from billing.price_note.
  function _money() {
    const c = (_billing && _billing.currency) || 'AUD';
    const sym = c === 'AUD' ? 'A$' : c === 'USD' ? 'US$' : c === 'EUR' ? '€' : (c + ' ');
    const note = _billing && _billing.price_note ? ' (' + _billing.price_note + ')' : '';
    return sym + Number(_price()).toFixed(2) + note;
  }
  const _unlockLabel = () => `Unlock — ${_money()} / ${_days()} days`;

  // Provider-agnostic: ask the server to create a checkout and redirect to
  // it. The server owns the provider (Creem now, anything later) — the page
  // never names one. Not logged in → sign in first (so the payment maps to a
  // user). No billing configured / any error → fall back to /login.
  async function openCheckout(plan) {
    const nx = encodeURIComponent(location.pathname + location.search);
    try {
      const r = await fetch('/api/billing/checkout', {
        method: 'POST', credentials: 'include',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(plan ? { plan } : {}),
      });
      if (r.status === 401) { window.location.href = '/signup?next=' + nx; return; }
      if (r.ok) {
        const d = await r.json();
        if (d && d.url) { window.location.href = d.url; return; }
      }
      // Signed in but checkout couldn't be created (5xx/other) — surface it
      // rather than bouncing to sign-in, which looks like a login problem.
      console.error('[billing] checkout failed', r.status);
      alert('Sorry — checkout is temporarily unavailable. Please try again in a moment.');
      return;
    } catch (e) {
      console.error('[billing] checkout error', e);
      alert('Sorry — checkout is temporarily unavailable. Please try again in a moment.');
    }
  }

  function _sym(cur) {
    return cur === 'AUD' ? 'A$' : cur === 'USD' ? 'US$' : cur === 'EUR' ? '€' : (cur || '') + ' ';
  }
  function _period(p) {
    if (p.mode === 'subscription') {
      if (p.days >= 360) return '/year';
      if (p.days >= 28) return '/month';
      return '/' + p.days + ' days';
    }
    return ' · ' + p.days + '-day pass';
  }
  // The 3-tier pricing chooser. One plan (or none) → straight to checkout.
  function openPricing() {
    const plans = (_billing && _billing.plans) || [];
    if (plans.length <= 1) { openCheckout(plans[0] && plans[0].key); return; }
    if (document.querySelector('.pricing-overlay')) return;
    const cur = (_billing && _billing.currency) || 'AUD';
    const tiers = plans.map((p, i) => `
      <button class="tier ${i === plans.length - 1 ? 'tier-best' : ''}" data-plan="${esc(p.key)}">
        ${i === plans.length - 1 ? '<span class="tier-badge">Best value</span>' : ''}
        <span class="tier-label">${esc(p.label || p.key)}</span>
        <span class="tier-price">${_sym(cur)}${Number(p.amount).toFixed(2)}<small>${esc(_period(p))}</small></span>
        <span class="tier-cta">Choose</span>
      </button>`).join('');
    const ov = document.createElement('div');
    ov.className = 'pricing-overlay';
    ov.innerHTML = `<div class="pricing-modal" role="dialog" aria-label="Choose a plan">
      <button class="pricing-close" aria-label="Close">&times;</button>
      <div class="pricing-head">Unlock every pick, all meetings</div>
      <div class="pricing-tiers">${tiers}</div>
      <div class="pricing-foot">Secure checkout via Stripe · cancel anytime</div>
    </div>`;
    ov.addEventListener('click', e => {
      if (e.target === ov || (e.target.closest && e.target.closest('.pricing-close'))) { ov.remove(); return; }
      const t = e.target.closest && e.target.closest('[data-plan]');
      if (t) { ov.remove(); openCheckout(t.dataset.plan); }
    });
    document.body.appendChild(ov);
  }

  // Delegated click handler — any [data-unlock] element opens the pricing chooser.
  // Idempotent: safe to call on every render.
  function bindUnlock() {
    if (document._pcUnlockBound) return;
    document._pcUnlockBound = true;
    document.addEventListener('click', e => {
      // Direct plan buy (e.g. the splash "5-Day Pass" button) → checkout.
      const buy = e.target.closest && e.target.closest('[data-buy]');
      if (buy) { e.preventDefault(); e.stopPropagation(); openCheckout(buy.dataset.buy); return; }
      const t = e.target.closest && e.target.closest('[data-unlock]');
      if (t) { e.preventDefault(); e.stopPropagation(); openPricing(); }
    }, true);
  }

  function paywallBanner(paywall) {
    if (!paywall || !paywall.active) return '';
    // Edge is subscription-tier (Monthly/Annual); Lounge/Hot Seat unlock with
    // any pass. Message + CTA adapt so a 5-day member sees the right thing.
    if (paywall.plan_required === 'subscription') {
      // Show the full plan cards inline (like the pre-publish splash).
      const up = plansUpsell(true);
      if (up) return up;
      return `<div class="paywall-banner" data-unlock>
        <div class="pb-txt"><b>The Edge is a Monthly &amp; Annual feature.</b> You're seeing 1 free race.</div>
        <a class="unlock-btn" href="/plans">See Monthly &amp; Annual</a>
      </div>`;
    }
    return `<div class="paywall-banner" data-unlock>
      <div class="pb-txt"><b>You're seeing 1 free race.</b> Unlock every pick, all meetings.</div>
      <button class="unlock-btn" data-unlock>${_unlockLabel()}</button>
    </div>`;
  }

  const _shortDate = iso => {
    try { return new Date(iso + 'T00:00:00+10:00').toLocaleDateString('en-AU', { weekday: 'short', day: 'numeric', month: 'short' }); }
    catch (e) { return iso || ''; }
  };
  // "Trophy for the day" — our biggest recent WIN, rendered as a full (settled)
  // race card so it carries all the normal info. A curated green highlight
  // (proof) shown to non-members even when there's no free upcoming pick.
  function trophyBanner(t) {
    if (!t || !t.horse_name) return '';
    const sp = t.sp != null ? '$' + Number(t.sp).toFixed(2) : '';
    const when = _shortDate(t.race_date);
    return `<div class="trophy-wrap" data-unlock>
      <div class="trophy-label">🏆 Recent winner — our top pick won at <b>${sp}</b>${when ? ' · ' + esc(when) : ''}</div>
      ${render(t, { trophy: true })}
      <button class="unlock-btn trophy-cta" data-unlock>Sign up — ${_money()} / ${_days()} days</button>
    </div>`;
  }

  function lockedCard(stub) {
    stub = stub || {};
    const venue = esc(stub.venue || stub.venue_name || stub.venue_display || '');
    const rno = stub.race_number != null ? stub.race_number
      : (stub.race_no != null ? stub.race_no : (stub.number != null ? stub.number : ''));
    const t = wallTime(stub.scheduled_time);
    const meta = [stub.distance ? stub.distance + 'm' : '', stub.field_size ? stub.field_size + ' runners' : '']
      .filter(Boolean).join(' · ');
    return `<div class="jcard pcard locked" data-unlock data-race="${esc(stub.race_id || '')}">
      <div class="mid">
        <div class="pk-venue">${venue}${rno !== '' ? ` <b>R${rno}</b>` : ''}</div>
        <div class="pk-horse locked-blur">Our pick</div>
        ${meta ? `<div class="pk-sub num">${meta}</div>` : ''}
        <div class="lock-cta"><span class="lock-ico">🔒</span> Members only</div>
      </div>
      <div class="right">
        ${t ? `<div class="when-top num">${t}</div>` : ''}
        <div class="lock-odds locked-blur">$0.00</div>
      </div>
    </div>`;
  }

  // Resilient JSON fetch. The /api/* proxy occasionally returns a transient
  // HTML error page (502/504) which blows up JSON.parse ("Unexpected token
  // '<'"). Retry once after a short pause before surfacing the error. Bounded
  // (single retry) — never a hammering loop.
  async function fetchJSON(url, init, retries) {
    retries = retries == null ? 1 : retries;
    let lastErr;
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const r = await fetch(url, init || {});
        const ct = r.headers.get('content-type') || '';
        if (r.ok && ct.indexOf('json') !== -1) return await r.json();
        lastErr = new Error('server busy (HTTP ' + r.status + ')');
      } catch (e) { lastErr = e; }
      if (attempt < retries) await new Promise(res => setTimeout(res, 600 + attempt * 600));
    }
    throw lastErr;
  }

  // Membership state (for the "picks being prepared" splash upsell). Resolved
  // once on load. { has: any active pass, days: latest plan length } — 5-day
  // pass = has:true, days:5; subscriber = days>=30.
  let _access = null;
  async function loadMembership() {
    try {
      const r = await fetch('/api/auth/me', { credentials: 'include' });
      if (r.ok) { const u = await r.json(); _access = { has: !!u.has_access, days: u.plan_days }; }
      else { _access = { has: false, days: null }; }
    } catch (e) { _access = { has: false, days: null }; }
    refreshSplashUpsell();
  }
  loadMembership();
  // Catch splashes that render after membership/billing already resolved.
  setTimeout(refreshSplashUpsell, 1500);
  setTimeout(refreshSplashUpsell, 3500);
  function isMember() { return !!(_access && _access.has); }
  function isSubscriber() { return !!(_access && _access.has && _access.days != null && _access.days >= 30); }
  // Active 5-day pass → eligible for the $9.90 credit on upgrade (mirrors the
  // backend _user_has_active_5day gate). Drives the in-app credit hint.
  function isFiveDayHolder() { return !!(_access && _access.has && _access.days === 5); }

  // The 3-tier plans grid (same content as /plans), rendered from live billing.
  const _SPLASH_FEATS = {
    base: ['Full form for every pick, all meetings', 'Lounge, Hot Seat & Listings', 'Sharp filter & value plays'],
    sub: ['<b>The Edge</b> — best picks, best odds', 'Every meeting, every race', 'Cancel anytime'],
    annual: ['<i>Exclusive NRL Predictions (70%+ accuracy, 2026)</i>'],
  };
  function _planPeriod(p) {
    if (p.mode === 'subscription') return p.days >= 360 ? '/year' : p.days >= 28 ? '/month' : '/' + p.days + ' days';
    return '';
  }
  function _planNote(p) {
    if (p.mode === 'subscription') return p.days >= 360 ? 'billed yearly' : 'billed monthly';
    return p.days + '-day pass · one-off';
  }
  function plansGrid(list) {
    const all = (_billing && _billing.plans) || [];
    if (!all.length) return '';
    list = list || all;
    if (!list.length) return '';
    const cur = (_billing && _billing.currency) || 'AUD';
    const monthly = all.find(p => p.mode === 'subscription' && p.days >= 28 && p.days < 360);
    const saveLine = p => {
      if (!(p.mode === 'subscription' && p.days >= 360 && monthly)) return '';
      const yearly = Number(monthly.amount) * 12;
      const pct = Math.round((1 - Number(p.amount) / yearly) * 100);
      return pct > 0 ? `<span class="fp-save"><span class="fp-was">${_sym(cur)}${yearly.toFixed(2)}/yr</span><span class="fp-off">Save ${pct}%</span></span>` : '';
    };
    return '<div class="fiq-plans">' + list.map((p, i) => {
      const best = i === list.length - 1 && list.length > 1;
      const isAnnual = p.mode === 'subscription' && p.days >= 360;
      const feats = _SPLASH_FEATS.base
        .concat(p.mode === 'subscription' ? _SPLASH_FEATS.sub : [])
        .concat(isAnnual ? _SPLASH_FEATS.annual : []);
      return `<div class="fp-tier${best ? ' best' : ''}">
        ${best ? '<span class="fp-badge">Best value</span>' : ''}
        <span class="fp-label">${esc(p.label || p.key)}</span>
        <span class="fp-price">${_sym(cur)}${Number(p.amount).toFixed(2)}${_planPeriod(p) ? `<small>${_planPeriod(p)}</small>` : ''}</span>
        ${saveLine(p)}
        ${(p.mode === 'subscription' && isFiveDayHolder()) ? `<span class="fp-credit">−${_sym(cur)}9.90 pass credit applied</span>` : ''}
        <span class="fp-note">${_planNote(p)}</span>
        <ul class="fp-feats">${feats.map(f => `<li>${f}</li>`).join('')}</ul>
        <button class="fp-cta" data-buy="${esc(p.key)}">Choose ${esc(p.label || p.key)}</button>
      </div>`;
    }).join('') + '</div>';
  }

  // Inline plan-cards upsell — used on the pre-publish splash, the Edge
  // subscription paywall, and the Playbook gate. Hidden for subscribers. A
  // 5-day pass holder sees only Monthly/Annual (upgrade pitch); non-members
  // see all three. `wide` lifts the width cap where there's no .today-gate
  // ancestor to widen it (Edge/Playbook).
  function plansUpsell(wide) {
    if (isSubscriber()) return '';
    const all = (_billing && _billing.plans) || [];
    if (!all.length) return '';
    const hasPass = isMember();              // active 5-day pass (not a subscriber)
    const list = hasPass ? all.filter(p => p.mode === 'subscription') : all;
    const grid = plansGrid(list);
    if (!grid) return '';
    const head = hasPass
      ? `<div class="tg-upsell-t">Get more from your membership</div>
         <div class="tg-upsell-s">You're on a 5-day pass. Go Monthly or Annual to unlock <b>the Edge</b> — and save with annual. Your <b>$9.90 pass credit</b> comes off your first payment.</div>`
      : `<div class="tg-upsell-t">🔓 Unlock every pick</div>
         <div class="tg-upsell-s">Choose a plan to unlock every pick, best odds, across every meeting.</div>`;
    return `<div class="tg-upsell${wide ? ' fp-wide' : ''}">${head}${grid}</div>`;
  }
  function splashUpsell() { return plansUpsell(false); }

  // Swap the splash upsell in place once membership/billing resolve, so the
  // right plans show without waiting for the page's slow (60s) poll.
  function refreshSplashUpsell() {
    var gates = document.querySelectorAll('.today-gate');
    for (var i = 0; i < gates.length; i++) {
      var gate = gates[i];
      var old = gate.querySelector('.tg-upsell');
      var html = splashUpsell();
      if (old) { old.outerHTML = html; }
      else if (html) { gate.insertAdjacentHTML('beforeend', html); }
      // Widen the splash so the plan cards can sit side by side.
      gate.classList.toggle('fiq-wide', !!gate.querySelector('.fiq-plans'));
    }
  }

  window.PickCard = {
    render, dualStat, winPlace, resultBlock, esc, wallTime, countdown,
    lockedCard, paywallBanner, trophyBanner, configureBilling, openCheckout, openPricing, bindUnlock,
    fetchJSON, isMember, isSubscriber, isFiveDayHolder, splashUpsell, plansUpsell, plansGrid, refreshSplashUpsell, loadMembership,
    fetchSparklines, isOpenRace, isHiddenRace, loadVenueBadges,
  };
  loadVenueBadges();   // warm the badged-venue set early so exceptions apply on first render
})();
