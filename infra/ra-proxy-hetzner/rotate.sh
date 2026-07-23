#!/usr/bin/env bash
# One-command RA-proxy droplet rotation.
#
# What it does:
#   1. Auto-bumps rotation_id in terraform.tfvars (YYYYMMDD, or YYYYMMDD-N for same-day)
#   2. terraform apply — new droplet with new IP, old destroyed
#   3. Waits for the new droplet's Caddy + FastAPI to be reachable over HTTPS
#   4. Verifies the healer script is well-formed and the new admin/cap-status
#      endpoint responds (proves the fixes made it into the built droplet)
#   5. Flips Railway RA_PROXY_URL to the new sslip hostname
#   6. Waits for Railway to redeploy, then hits an admin endpoint end-to-end
#      to confirm proxy → Railway path is healthy
#
# Requirements:
#   - terraform CLI on PATH
#   - railway CLI on PATH, logged in, linked to the horse-prediction service
#   - CRON_SECRET env var set (for the end-to-end verification step)
#
# Usage:
#   ./rotate.sh                # do it
#   ./rotate.sh --plan-only    # show terraform plan, don't apply
#   ./rotate.sh --skip-railway # rotate droplet only, don't flip RA_PROXY_URL
#
# Safety:
#   - Fails fast on any error
#   - Prints the new IP and asks for confirmation before Railway update
#     (unless --yes is passed)

set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MODULE_DIR"

PLAN_ONLY=0
SKIP_RAILWAY=0
AUTO_YES=0
for arg in "$@"; do
  case "$arg" in
    --plan-only)   PLAN_ONLY=1 ;;
    --skip-railway) SKIP_RAILWAY=1 ;;
    --yes|-y)      AUTO_YES=1 ;;
    -h|--help)
      sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'
      exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\033[1;36m[rotate]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[rotate]\033[0m %s\n' "$*" >&2; }

# --- 1. Bump rotation_id ---------------------------------------------------
[[ -f terraform.tfvars ]] || { err "terraform.tfvars missing"; exit 1; }
today=$(date +%Y%m%d)
current=$(grep -oE 'rotation_id[[:space:]]*=[[:space:]]*"[^"]+"' terraform.tfvars | sed -E 's/.*"([^"]+)".*/\1/')
if [[ -z "$current" ]]; then
  err "could not find rotation_id in terraform.tfvars"; exit 1
fi

if [[ "$current" == "$today" ]]; then
  next="${today}-2"
elif [[ "$current" == "${today}-"* ]]; then
  suffix="${current#${today}-}"
  # Non-numeric suffixes (e.g. 20260721-fsn1) can't be arithmetic-bumped
  # and 'set -u' turns $((fsn1+1)) into an unbound-variable crash. Reset
  # to a plain -2 in that case; the user can hand-set a fancier tag later.
  if [[ "$suffix" =~ ^[0-9]+$ ]]; then
    next="${today}-$((suffix+1))"
  else
    next="${today}-2"
  fi
else
  next="$today"
fi

log "rotation_id: $current → $next"
if [[ $PLAN_ONLY -eq 0 ]]; then
  sed -i.bak -E "s/(rotation_id[[:space:]]*=[[:space:]]*\")[^\"]+(\")/\1${next}\2/" terraform.tfvars
  rm -f terraform.tfvars.bak
fi

# --- 2. Terraform plan/apply -----------------------------------------------
# NOTE: hcloud_server.name is an IN-PLACE update, not force-new. Bumping
# rotation_id alone only renames the box — same IP, so RA's WAF block on
# the old IP persists (this silently defeated the 2026-07-22 rotation).
# Force-replace the server so every rotation genuinely gets a NEW IP.
log "terraform plan (force-replacing ra_proxy for a new IP)"
terraform plan -replace="hcloud_server.ra_proxy" -out=rotate.plan

if [[ $PLAN_ONLY -eq 1 ]]; then
  log "plan-only mode — stopping here. Plan saved as rotate.plan."
  exit 0
fi

if [[ $AUTO_YES -eq 0 ]]; then
  read -rp "Apply this plan? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { log "aborted"; exit 0; }
fi

log "terraform apply"
terraform apply rotate.plan
rm -f rotate.plan

NEW_IP=$(terraform output -raw droplet_ip)
NEW_URL=$(terraform output -raw proxy_url)
log "new droplet: $NEW_IP ($NEW_URL)"

# --- 3. Wait for HTTPS reachability ---------------------------------------
log "waiting for cloud-init + Caddy TLS (up to 5 min)…"
deadline=$(( $(date +%s) + 300 ))
while : ; do
  if curl -sf --max-time 8 "$NEW_URL/health" >/dev/null 2>&1; then
    log "proxy reachable"
    break
  fi
  if [[ $(date +%s) -gt $deadline ]]; then
    err "timed out waiting for $NEW_URL/health — check droplet console"
    exit 1
  fi
  sleep 10
done

# --- 4. Verify source-of-truth fixes landed on the droplet ----------------
log "verifying source-of-truth fixes on droplet"

# Healer script must not contain the templating bug ($$)
if ssh -o StrictHostKeyChecking=accept-new "root@${NEW_IP}" \
     "grep -q '\$\$' /usr/local/bin/ra-proxy-healthcheck.sh" 2>/dev/null; then
  err "healer script still contains \$\$ — template bug regressed"
  exit 1
fi
log "  ✓ healer script clean"

# admin/cap-status endpoint must respond
CAP_STATUS=$(ssh "root@${NEW_IP}" \
  'curl -s http://127.0.0.1:8000/admin/cap-status -H "x-proxy-secret: $(grep PROXY_SECRET /etc/ra-proxy.env | cut -d= -f2-)"' \
  2>/dev/null || echo "")
if ! echo "$CAP_STATUS" | grep -q '"daily_cap"'; then
  err "admin/cap-status not responding — proxy.py changes may not have landed"
  err "response: $CAP_STATUS"
  exit 1
fi
log "  ✓ admin/cap-status endpoint working"
log "  cap-status: $CAP_STATUS"

if [[ $SKIP_RAILWAY -eq 1 ]]; then
  log "--skip-railway: NOT updating Railway env var. Do it manually with:"
  log "  railway variables set RA_PROXY_URL=$NEW_URL"
  exit 0
fi

# --- 5. Flip Railway env var ----------------------------------------------
if [[ $AUTO_YES -eq 0 ]]; then
  read -rp "Flip Railway RA_PROXY_URL to $NEW_URL? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { log "aborted before Railway flip"; exit 0; }
fi

log "railway variables set RA_PROXY_URL=$NEW_URL"
railway variables set "RA_PROXY_URL=$NEW_URL"

# --- 6. End-to-end verification -------------------------------------------
log "waiting 90s for Railway redeploy…"
sleep 90

if [[ -z "${CRON_SECRET:-}" ]]; then
  log "CRON_SECRET not set — skipping end-to-end seed test"
else
  log "hitting Railway → proxy → RA (probe-ra-results)"
  probe=$(curl -s --max-time 60 \
    -H "x-cron-secret: $CRON_SECRET" \
    "https://web-production-dec62.up.railway.app/api/admin/probe-ra-results?date=$(date +%Y-%m-%d)&venue_code=TAM")
  if echo "$probe" | grep -q '"status": *200'; then
    log "  ✓ Railway → new proxy → RA path healthy"
  else
    err "  ⚠ probe did not report 200 — check Railway logs. First 500 chars:"
    err "  ${probe:0:500}"
  fi
fi

log "done. new droplet: $NEW_IP  $NEW_URL"
