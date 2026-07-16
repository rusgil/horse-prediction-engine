#!/usr/bin/env bash
# ------------------------------------------------------------------
# RA proxy droplet migration
#
# Rotates the DigitalOcean droplet that fronts Racing Australia when
# RA WAF-blocks the current outbound IP. Same playbook as 2026-07-14.
#
# Prereqs (install once, then re-run this script whenever needed):
#   1. Homebrew: brew install doctl
#   2. Get a DigitalOcean API token with read+write scope:
#      https://cloud.digitalocean.com/account/api/tokens
#   3. Authenticate:  doctl auth init  (paste the token when prompted)
#   4. Add your SSH public key to DigitalOcean (Settings → Security)
#   5. Ensure your Railway CLI is installed (`brew install railway`)
#      and you're logged in (`railway login`), or set the two Railway
#      env vars manually via the dashboard.
#
# Usage:  ./scripts/migrate-ra-proxy.sh [--dry-run]
# ------------------------------------------------------------------
set -euo pipefail

DRY_RUN=false
if [[ "${1-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "[dry-run] no side effects will occur"
fi

# -- 0. sanity --------------------------------------------------------
command -v doctl >/dev/null || {
  echo "❌  doctl not installed. Run: brew install doctl" >&2; exit 1; }
command -v jq >/dev/null || {
  echo "❌  jq not installed. Run: brew install jq" >&2; exit 1; }

# -- 1. locate the current droplet ------------------------------------
# We identify by the droplet's tag 'ra-proxy'. If your droplet doesn't
# have that tag yet, tag it once via:
#   doctl compute droplet tag <droplet-id> --tag-name ra-proxy
CURRENT_ID=$(doctl compute droplet list --tag-name ra-proxy --format ID --no-header | head -1)
if [[ -z "${CURRENT_ID}" ]]; then
  echo "❌  no droplet with tag 'ra-proxy' found" >&2; exit 1
fi

CURRENT_META=$(doctl compute droplet get "${CURRENT_ID}" --format Name,PublicIPv4,Region,Size --no-header)
CURRENT_NAME=$(echo "${CURRENT_META}" | awk '{print $1}')
CURRENT_IP=$(echo "${CURRENT_META}" | awk '{print $2}')
CURRENT_REGION=$(echo "${CURRENT_META}" | awk '{print $3}')
CURRENT_SIZE=$(echo "${CURRENT_META}" | awk '{print $4}')

echo "▸ current droplet"
echo "    id:      ${CURRENT_ID}"
echo "    name:    ${CURRENT_NAME}"
echo "    ip:      ${CURRENT_IP}"
echo "    region:  ${CURRENT_REGION}"
echo "    size:    ${CURRENT_SIZE}"

# -- 2. snapshot ------------------------------------------------------
STAMP=$(date -u +%Y%m%d-%H%M)
SNAP_NAME="ra-proxy-${STAMP}"
NEW_NAME="ra-proxy-${STAMP}"

if $DRY_RUN; then
  echo "[dry-run] would create snapshot: ${SNAP_NAME}"
else
  echo "▸ creating snapshot ${SNAP_NAME} (2-6 min)…"
  doctl compute droplet-action snapshot "${CURRENT_ID}" --snapshot-name "${SNAP_NAME}" --wait
fi

# -- 3. find the snapshot id -----------------------------------------
if ! $DRY_RUN; then
  SNAP_ID=$(doctl compute snapshot list --format ID,Name --no-header \
    | awk -v n="${SNAP_NAME}" '$2==n {print $1}' | head -1)
  if [[ -z "${SNAP_ID}" ]]; then
    echo "❌  couldn't locate freshly created snapshot ${SNAP_NAME}" >&2; exit 1
  fi
  echo "  snapshot id: ${SNAP_ID}"
fi

# -- 4. pick an SSH key ----------------------------------------------
SSH_KEY_ID=$(doctl compute ssh-key list --format ID --no-header | head -1)
if [[ -z "${SSH_KEY_ID}" ]]; then
  echo "❌  no SSH key registered with DO. Add one at https://cloud.digitalocean.com/account/security" >&2; exit 1
fi
echo "  using SSH key id: ${SSH_KEY_ID}"

# -- 5. create the new droplet ---------------------------------------
if $DRY_RUN; then
  echo "[dry-run] would create droplet ${NEW_NAME} in ${CURRENT_REGION} (${CURRENT_SIZE})"
  NEW_IP="0.0.0.0"
else
  echo "▸ creating new droplet ${NEW_NAME} (2-3 min)…"
  NEW_DROPLET_JSON=$(doctl compute droplet create "${NEW_NAME}" \
    --image "${SNAP_ID}" \
    --size "${CURRENT_SIZE}" \
    --region "${CURRENT_REGION}" \
    --ssh-keys "${SSH_KEY_ID}" \
    --tag-name ra-proxy \
    --wait \
    --output json)
  NEW_IP=$(echo "${NEW_DROPLET_JSON}" | jq -r '.[0].networks.v4[] | select(.type=="public") | .ip_address')
  NEW_ID=$(echo "${NEW_DROPLET_JSON}" | jq -r '.[0].id')
  echo "  new droplet id: ${NEW_ID}"
  echo "  new droplet ip: ${NEW_IP}"
fi

NEW_HOST_SLIP="${NEW_IP//./-}.sslip.io"
NEW_URL="https://${NEW_HOST_SLIP}"

# -- 6. fix Caddy hostname on new droplet ----------------------------
# Caddy on the snapshot still has the old hostname baked in for TLS.
# Rewrite Caddyfile and reload — a couple of seconds of downtime while
# LetsEncrypt issues a cert for the new sslip.io host.
OLD_HOST_SLIP="${CURRENT_IP//./-}.sslip.io"
if $DRY_RUN; then
  echo "[dry-run] would rewrite Caddyfile on new droplet: ${OLD_HOST_SLIP} → ${NEW_HOST_SLIP}"
else
  echo "▸ rewriting Caddyfile hostname on new droplet (~30s)…"
  # Small delay so cloud-init finishes and SSH is available
  sleep 30
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@${NEW_IP}" \
    "sed -i 's/${OLD_HOST_SLIP}/${NEW_HOST_SLIP}/g' /etc/caddy/Caddyfile && systemctl reload caddy"
  echo "▸ verifying /health on new droplet…"
  for i in 1 2 3 4 5; do
    sleep 6
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "${NEW_URL}/health" || echo 000)
    echo "    attempt ${i}: HTTP ${CODE}"
    if [[ "${CODE}" == "200" ]]; then
      break
    fi
  done
fi

# -- 7. update Railway env vars --------------------------------------
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "▸ next steps (manual — Railway API token varies per project):"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "1. Update Railway env var (via dashboard or CLI):"
echo "     railway variables set RA_PROXY_URL=${NEW_URL}"
echo ""
echo "2. Railway auto-redeploys on env change (~90s). Watch:"
echo "     railway logs --deployment"
echo ""
echo "3. Once redeployed, verify results seeding works:"
echo "     curl -s -X POST -H 'x-cron-secret: <SECRET>' \\"
echo "       https://web-production-dec62.up.railway.app/api/admin/seed-ra-results/\$(date +%Y-%m-%d)"
echo ""
echo "4. Wait 24-48h to confirm stability. Then destroy the old droplet:"
echo "     doctl compute droplet delete ${CURRENT_ID}"
echo ""
echo "5. Optional cleanup — remove old snapshots >30 days:"
echo "     doctl compute snapshot list --format ID,Name,CreatedAt"
echo ""
echo "▸ new droplet URL: ${NEW_URL}"
