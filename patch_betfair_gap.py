"""
Patch historical DB records for the Punters outage gap period using
Betfair historical streaming data from data.tar.

For each settled AU WIN/PLACE market in the tar:
  1. Patches starting_price (BSP) onto historical_results rows where NULL
  2. Inserts odds_snapshots rows from the full LTP history
  3. Patches best_available_odds onto runner_prediction_history where NULL/0

Usage:
    # Dry run — show what would be patched, no DB writes
    python patch_betfair_gap.py --dry-run

    # Patch a specific date range
    python patch_betfair_gap.py --from 2026-04-15 --to 2026-05-31

    # Patch just BSP (skip LTP snapshots)
    python patch_betfair_gap.py --bsp-only

Patches via the Railway admin API (no direct DB access needed):
    POST /api/admin/patch-betfair-bsp
"""
from __future__ import annotations

import argparse
import bz2
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

_API_URL = "https://web-production-dec62.up.railway.app"
_CRON_SECRET = "horse2026secretJH"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TAR_PATH = Path("/Users/russellgilbert/funkyiq/data.tar")
AEST = timezone(timedelta(hours=10))

# ── Venue name normalisation ──────────────────────────────────────────────────

_DATE_SUFFIX = re.compile(
    r'\s+\d+\w*\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*$',
    re.IGNORECASE
)

def _slugify(name: str) -> str:
    clean = re.sub(r'\s*\([^)]+\)', '', name).strip()
    clean = _DATE_SUFFIX.sub('', clean).strip()
    return re.sub(r'[^a-z0-9]+', '-', clean.lower()).strip('-')


def _strip_barrier(name: str) -> str:
    """'1. Dark Fox' → 'Dark Fox'"""
    return re.sub(r'^\d+\.\s*', '', name).strip()


def _race_num_from_name(market_name: str) -> int | None:
    m = re.match(r'[Rr](?:ace\s*)?(\d+)', market_name)
    return int(m.group(1)) if m else None


# ── Betfair market parser ─────────────────────────────────────────────────────

def parse_win_market(raw_bz2: bytes) -> dict | None:
    """
    Parse a WIN market bz2 stream.
    Returns dict with venue, date_aest, race_number, runners (name+bsp+status),
    and ltp_history (runner_name → [(ts_ms, price)]).
    Returns None if market not settled.
    """
    try:
        data = bz2.decompress(raw_bz2).decode('utf-8', errors='replace')
    except Exception:
        return None

    lines = [l for l in data.split('\n') if l.strip()]
    if not lines:
        return None

    # First line has initial market definition
    try:
        first = json.loads(lines[0])
    except Exception:
        return None

    mc0 = (first.get('mc') or [{}])[0]
    md0 = mc0.get('marketDefinition', {})

    if md0.get('countryCode') != 'AU':
        return None
    if md0.get('marketType') != 'WIN':
        return None

    race_num = _race_num_from_name(md0.get('marketName') or '')
    market_time_str = md0.get('marketTime', '')
    try:
        market_time_utc = datetime.fromisoformat(market_time_str.replace('Z', '+00:00'))
        market_time_aest = market_time_utc.astimezone(AEST)
        date_aest = market_time_aest.date().isoformat()
        market_ms = market_time_utc.timestamp() * 1000
    except Exception:
        return None

    venue_raw = md0.get('venue') or md0.get('eventName', '')
    venue_slug = _slugify(venue_raw)

    # Build runner id→name map and LTP history from all messages
    id_to_name: dict[int, str] = {}
    ltp_history: dict[int, list] = defaultdict(list)  # runner_id → [(ts_ms, price)]
    final_md: dict | None = None

    for line in lines:
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get('op') != 'mcm':
            continue
        pt = msg.get('pt', 0)
        for mc in msg.get('mc', []):
            md = mc.get('marketDefinition')
            if md:
                for runner in md.get('runners', []):
                    rid = runner.get('id')
                    name = runner.get('name', '')
                    if rid and name:
                        id_to_name[rid] = name
                if md.get('status') == 'CLOSED' and md.get('bspReconciled'):
                    final_md = md
            for rc in mc.get('rc', []):
                rid = rc.get('id')
                ltp = rc.get('ltp')
                if rid and ltp and pt:
                    ltp_history[rid].append((pt, float(ltp)))

    if not final_md:
        return None  # market not settled

    runners = []
    for r in final_md.get('runners', []):
        rid = r.get('id')
        name = id_to_name.get(rid) or r.get('name', '')
        clean_name = _strip_barrier(name)
        status = r.get('status', '')
        bsp = r.get('bsp')
        runners.append({
            'runner_id': rid,
            'name_raw': name,
            'name': clean_name,
            'status': status,   # WINNER / LOSER / REMOVED
            'bsp': bsp,
        })

    # Build named LTP history (strip barrier prefix from names)
    named_ltp: dict[str, list] = {}
    for rid, hist in ltp_history.items():
        name = _strip_barrier(id_to_name.get(rid, ''))
        if name:
            named_ltp[name] = sorted(hist, key=lambda x: x[0])

    return {
        'venue_raw': venue_raw,
        'venue_slug': venue_slug,
        'date_aest': date_aest,
        'race_number': race_num,
        'market_time_ms': market_ms,
        'settled_time': final_md.get('settledTime'),
        'runners': runners,
        'ltp_history': named_ltp,
    }


def parse_place_market(raw_bz2: bytes) -> dict | None:
    """
    Parse a PLACE market bz2 to extract who placed (WINNER status = placed).
    Returns dict with date_aest, venue_slug, race_number, placers (list of names).
    """
    try:
        data = bz2.decompress(raw_bz2).decode('utf-8', errors='replace')
    except Exception:
        return None

    lines = [l for l in data.split('\n') if l.strip()]
    if not lines:
        return None

    final_md = None
    id_to_name: dict[int, str] = {}
    market_time_str = ''
    venue_raw = ''
    race_num = None

    for line in lines:
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get('op') != 'mcm':
            continue
        for mc in msg.get('mc', []):
            md = mc.get('marketDefinition')
            if md:
                if md.get('countryCode') != 'AU':
                    return None
                if md.get('marketType') != 'PLACE':
                    return None
                if not market_time_str:
                    market_time_str = md.get('marketTime', '')
                    venue_raw = md.get('venue') or md.get('eventName', '')
                    race_num = _race_num_from_name(md.get('marketName') or '')
                for runner in md.get('runners', []):
                    rid = runner.get('id')
                    name = runner.get('name', '')
                    if rid and name:
                        id_to_name[rid] = name
                if md.get('status') == 'CLOSED' and md.get('bspReconciled'):
                    final_md = md

    if not final_md or not market_time_str:
        return None

    try:
        market_time_utc = datetime.fromisoformat(market_time_str.replace('Z', '+00:00'))
        date_aest = market_time_utc.astimezone(AEST).date().isoformat()
    except Exception:
        return None

    placers = []
    for r in final_md.get('runners', []):
        if r.get('status') == 'WINNER':
            name = _strip_barrier(id_to_name.get(r.get('id'), r.get('name', '')))
            if name:
                placers.append(name)

    try:
        market_ms = datetime.fromisoformat(market_time_str.replace('Z', '+00:00')).timestamp() * 1000
    except Exception:
        market_ms = 0

    return {
        'venue_slug': _slugify(venue_raw),
        'date_aest': date_aest,
        'race_number': race_num,
        'market_time_ms': market_ms,
        'placers': placers,
    }


# ── Tar scanning ──────────────────────────────────────────────────────────────

def scan_tar(date_from: str, date_to: str) -> dict[str, list[str]]:
    """
    Scan tar for AU WIN and PLACE market files in the date range.
    Returns dict: 'win' → [path, ...], 'place' → [path, ...]
    Uses tar -tf (fast, no extraction).
    """
    log.info("Scanning %s for files between %s and %s ...", TAR_PATH, date_from, date_to)
    result = subprocess.run(
        ['tar', '-tf', str(TAR_PATH)],
        capture_output=True, text=True, check=True
    )

    # Convert date range to month/day sets for quick filtering
    from_dt = datetime.fromisoformat(date_from)
    to_dt = datetime.fromisoformat(date_to)

    # Build set of BASIC/YYYY/Mon/DD prefixes in range
    valid_prefixes: set[str] = set()
    current = from_dt
    while current <= to_dt:
        mon = current.strftime('%b')
        prefix = f"BASIC/{current.year}/{mon}/{current.day}/"
        valid_prefixes.add(prefix)
        current += timedelta(days=1)

    # We need to check if a file path's date prefix is in range
    # File format: BASIC/2026/May/15/EVENT/MARKET.bz2
    market_files = {'win': [], 'place': []}

    for path in result.stdout.splitlines():
        if not path.endswith('.bz2'):
            continue
        # Skip event-level files (event_id.bz2, not 1.XXXXXXX.bz2)
        fname = path.split('/')[-1]
        if not fname.startswith('1.'):
            continue

        # Check date prefix
        matched = any(path.startswith(p) for p in valid_prefixes)
        if not matched:
            continue

        # We'll check WIN vs PLACE when we extract (too slow to check here)
        # Just collect all individual market files
        market_files['win'].append(path)  # we'll filter by type after extraction

    log.info("Found %d candidate market files in date range", len(market_files['win']))
    return market_files['win']


def extract_file(tar_path: Path, inner_path: str) -> bytes | None:
    """Extract a single file from the tar and return its raw bytes."""
    result = subprocess.run(
        ['tar', '-xOf', str(tar_path), inner_path],
        capture_output=True
    )
    return result.stdout if result.returncode == 0 else None


# ── Main processing ───────────────────────────────────────────────────────────

def build_race_map(file_paths: list[str], date_from: str, date_to: str) -> dict:
    """
    Extract and parse all market files. Race numbers are assigned by sorting WIN
    markets within each (date, venue) group by market_time_ms (BASIC tier has no
    market names, so time-ordering is the only reliable approach).

    Returns: race_map[date_aest][venue_slug][race_number] = {
        'win': parsed_win_dict,
        'placers': [names...],
    }
    """
    log.info("Parsing %d candidate market files ...", len(file_paths))

    # Intermediate storage before race-number assignment
    # key: (date_aest, venue_slug) → list of parsed WIN dicts (unsorted)
    wins_by_meeting: dict[tuple, list] = defaultdict(list)
    # key: (date_aest, venue_slug, market_time_ms) → list of placer names
    placers_by_time: dict[tuple, list] = defaultdict(list)

    win_count = place_count = skip_count = 0

    for i, path in enumerate(file_paths):
        if i % 500 == 0 and i > 0:
            log.info("  Processed %d / %d files (wins=%d, places=%d, skipped=%d)",
                     i, len(file_paths), win_count, place_count, skip_count)

        raw = extract_file(TAR_PATH, path)
        if not raw:
            skip_count += 1
            continue

        # Try WIN first
        win = parse_win_market(raw)
        if win:
            key = (win['date_aest'], win['venue_slug'])
            wins_by_meeting[key].append(win)
            win_count += 1
            continue

        # Try PLACE
        place = parse_place_market(raw)
        if place:
            key = (place['date_aest'], place['venue_slug'], place.get('market_time_ms', 0))
            placers_by_time[key] = place['placers']
            place_count += 1
            continue

        skip_count += 1

    log.info("Parsed: %d WIN markets, %d PLACE markets, %d skipped",
             win_count, place_count, skip_count)

    # Assign race numbers by sorting WIN markets within each meeting by start time
    race_map: dict = defaultdict(lambda: defaultdict(dict))
    for (date_aest, venue_slug), wins in wins_by_meeting.items():
        sorted_wins = sorted(wins, key=lambda w: w['market_time_ms'])
        for race_num, win_data in enumerate(sorted_wins, start=1):
            # Find matching PLACE market by time (within 2 min)
            t_ms = win_data['market_time_ms']
            placers = []
            for (pd, pv, pt_ms), plist in placers_by_time.items():
                if pd == date_aest and pv == venue_slug and abs(pt_ms - t_ms) < 120_000:
                    placers = plist
                    break
            race_map[date_aest][venue_slug][race_num] = {
                'win': win_data,
                'placers': placers,
            }

    return race_map


# ── DB patching ───────────────────────────────────────────────────────────────

def patch_via_api(race_map: dict, dry_run: bool, bsp_only: bool,
                  api_url: str, secret: str, batch_size: int = 50) -> None:
    """POST patch data to the Railway admin endpoint in batches."""
    import urllib.request
    import urllib.error

    endpoint = f"{api_url.rstrip('/')}/api/admin/patch-betfair-bsp"
    headers = {
        'Content-Type': 'application/json',
        'x-cron-secret': secret,
    }

    total_bsp = total_pred = total_snaps = total_skipped = 0
    batch: list[dict] = []

    def _flush(b: list[dict]) -> None:
        nonlocal total_bsp, total_pred, total_snaps, total_skipped
        if dry_run:
            runners_total = sum(len(r.get('runners', [])) for r in b)
            log.info("DRY RUN: would POST batch of %d races / %d runners", len(b), runners_total)
            return
        body = json.dumps({'races': b}).encode()
        req = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                total_bsp += result.get('bsp_patched', 0)
                total_pred += result.get('pred_patched', 0)
                total_snaps += result.get('snap_inserted', 0)
                total_skipped += result.get('bsp_skipped', 0)
        except urllib.error.HTTPError as e:
            log.error("API error %d: %s", e.code, e.read()[:200])

    for date_aest, venues in sorted(race_map.items()):
        for venue_slug, races in venues.items():
            for race_num, data in races.items():
                win_data = data.get('win')
                if not win_data:
                    continue

                race_id = f"{date_aest}_{venue_slug}_R{race_num}"
                runners_out = []

                for runner in win_data['runners']:
                    if runner['status'] == 'REMOVED':
                        continue
                    bsp = runner.get('bsp')
                    name = runner['name']
                    if not bsp:
                        continue

                    snapshots = []
                    if not bsp_only:
                        market_ms = win_data['market_time_ms']
                        for ts_ms, price in win_data['ltp_history'].get(name, []):
                            mtj = round((market_ms - ts_ms) / 60_000, 1)
                            snap_time = datetime.fromtimestamp(
                                ts_ms / 1000, tz=timezone.utc
                            ).strftime('%Y-%m-%dT%H:%M:%SZ')
                            snapshots.append({
                                'minutes_to_jump': mtj,
                                'snapshotted_at': snap_time,
                                'win_odds': price,
                            })

                    runners_out.append({'name': name, 'bsp': bsp, 'snapshots': snapshots})

                if runners_out:
                    batch.append({'race_id': race_id, 'runners': runners_out})

                if len(batch) >= batch_size:
                    _flush(batch)
                    batch = []

    if batch:
        _flush(batch)

    log.info("Patch complete — BSP rows: %d, pred rows: %d, snapshots: %d, skipped: %d",
             total_bsp, total_pred, total_snaps, total_skipped)
    if dry_run:
        log.info("DRY RUN — no changes written")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Patch DB with Betfair historical data")
    parser.add_argument('--from', dest='date_from', default='2026-04-15',
                        help='Start date (inclusive), e.g. 2026-04-15')
    parser.add_argument('--to', dest='date_to', default='2026-05-31',
                        help='End date (inclusive), e.g. 2026-05-31')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show counts without writing to DB')
    parser.add_argument('--bsp-only', action='store_true',
                        help='Only patch BSP; skip LTP snapshot insertion')
    args = parser.parse_args()

    if not TAR_PATH.exists():
        log.error("Tar file not found: %s", TAR_PATH)
        sys.exit(1)

    # Step 1: scan tar for candidates
    candidate_paths = scan_tar(args.date_from, args.date_to)

    # Step 2: parse markets
    race_map = build_race_map(candidate_paths, args.date_from, args.date_to)

    total_races = sum(
        len(races)
        for venues in race_map.values()
        for races in venues.values()
    )
    log.info("Parsed %d AU WIN markets across %d race days", total_races, len(race_map))

    # Step 3: POST patch data to Railway API
    patch_via_api(
        race_map,
        dry_run=args.dry_run,
        bsp_only=args.bsp_only,
        api_url=_API_URL,
        secret=_CRON_SECRET,
    )


if __name__ == '__main__':
    main()
