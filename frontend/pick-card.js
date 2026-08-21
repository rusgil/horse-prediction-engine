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
  function dualStat(win_pct, place_pct, accuracy) {
    if (win_pct == null) return '';
    const f = v => (+Number(v).toFixed(1));
    let out = `<div class="ds"><span class="ds-num num ${win_pct >= 30 ? 'v-win' : ''}">${f(win_pct)}%</span><span class="ds-lbl">win prediction</span></div>`;
    if (place_pct != null) out += `<div class="ds-sep"></div><div class="ds"><span class="ds-num num ${place_pct >= 50 ? 'v-plc' : ''}">${f(place_pct)}%</span><span class="ds-lbl">place prediction</span></div>`;
    if (accuracy != null) out += `<div class="ds-sep"></div><div class="ds"><span class="ds-num num">${f(accuracy)}%</span><span class="ds-lbl">model accuracy at this level</span></div>`;
    return `<div class="dual-stat-row">${out}</div>`;
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
    return `<div class="jcard pcard ${outcome} ${!isPast && pick.confidence_tier === 'hot' ? 'hot' : ''} ${!isPast && pick.is_premium ? 'prem' : ''}" style="${dim}" data-open="${esc(pick.race_id)}" data-horse="${esc(pick.horse_name)}">
      <div class="when">
        ${isPast
          ? `<div class="clock num">${wallTime(pick.scheduled_time)}</div><div class="cd">${hasResult ? (res.winner ? '<span class="res win">✓</span>' : res.placed ? '<span class="res place">✓' + (posLbl ? ' ' + posLbl : '') + '</span>' : '<span class="res miss">✗</span>') : ''}</div>`
          : `<div class="cd big ${cd.urg || ''}">${cd.label || ''}</div><div class="wt num">${wallTime(pick.scheduled_time)}</div>`}
      </div>
      <div class="mid">
        <div class="pk-venue">${esc(pick.venue)} <b>R${pick.race_number}</b>${badge}</div>
        <div class="pk-horse">${esc(pick.horse_name)}${streak}${!isPast ? placeChip(pick) : ''}</div>
        <div class="pk-sub num">${[pick.distance ? pick.distance + 'm' : '', pick.barrier ? 'B' + pick.barrier : '', pick.weight ? pick.weight + 'kg' : ''].filter(Boolean).join(' · ')}${l10}</div>
        <div class="pk-sub">${jt}${pick.days_off != null && pick.days_off < 999 ? ` · ${pick.days_off}d off` : ''}</div>
        <div class="pk-chips">${tierChip(pick)}${!isPast && pick.is_premium ? '<span class="chip prem">💎 PREMIUM</span>' : ''}${pick.is_sharp ? '<span class="chip sharp">🎯 SHARP</span>' : ''}${!isPast ? favChip(pick) : ''}${!isPast ? exoticChips(pick) : ''}${pick.sportsbet_available === false ? '<span class="chip nosb">🚫 Not on Sportsbet</span>' : ''}</div>
      </div>
      <div class="right">
        ${displayOdds ? `<div class="odds-pill ${oddsCls}"><b class="num">$${(+displayOdds).toFixed(2)}</b>${oddsLbl}</div>` : '<div class="odds-pill">TBA</div>'}
        ${settledRes && res.winner && res.place_odds ? `<div class="odds-pill"><b class="num" style="font-size:1rem;color:var(--blue)">$${(+res.place_odds).toFixed(2)}</b>place</div>` : ''}
        ${!isPast && ctx.showSpark ? `<div class="spark-slot" id="spark-${pick.race_id.replace(/[^a-z0-9]/gi, '-')}"></div>` : ''}
      </div>
      ${dualStat(pick.win_pct, pick.place_pct, pick.accuracy != null ? pick.accuracy : calibratedRate(pick.win_pct))}
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

  window.PickCard = { render, dualStat, resultBlock, esc, wallTime, countdown };
})();
