# ra-proxy (Hetzner Cloud port)

Same reverse-proxy design as the sibling `ra-proxy/` module (DigitalOcean),
ported to Hetzner Cloud. Built because RA WAF-fingerprinted the entire
DO netblock on 2026-07-17.

**Why Hetzner:** completely different ASN than DO, low scraper reputation,
Singapore location keeps latency to Australia acceptable (~90ms extra),
runs a proxy for AUD ~$7/month.

**Same design:**
- One Ubuntu 24.04 server per rotation
- Caddy in front for LetsEncrypt TLS on an sslip.io hostname
- FastAPI proxy behind, using `curl_cffi` to match a real Chrome TLS
  fingerprint so RA's WAF can't detect us as a bot
- systemd unit + 2-min self-heal timer
- ufw / Hetzner Cloud firewall locking down to 22/80/443/ICMP

## First-time setup

1. Sign up at https://console.hetzner.cloud and create a project.
2. Add your SSH public key via console (Security → SSH Keys). Give it
   a friendly name — you'll paste that name into `terraform.tfvars`.
3. Generate an API token: Security → API Tokens → New. Give it
   Read & Write access. Copy the token — you can't view it again after
   creation.
4. `cp terraform.tfvars.example terraform.tfvars` and fill in:
   - `hcloud_token` — the API token from step 3
   - `proxy_secret` — must match Railway's `RA_PROXY_SECRET` env var
   - `ssh_key_names` — the name(s) from step 2

## First deploy

```bash
cd infra/ra-proxy-hetzner
terraform init
export CRON_SECRET='<your CRON_SECRET>'
./rotate.sh --yes
```

The wrapper auto-bumps `rotation_id`, applies, waits for HTTPS, verifies
the healer + `/admin/cap-status` on the new server, flips
`RA_PROXY_URL` on Railway, then probes end-to-end.

## Rotation

Same as first deploy. `./rotate.sh --yes` on any subsequent run creates
a new server with a new IP, destroys the old one, and reflows Railway.

`./rotate.sh --plan-only` shows the plan without applying — useful for
sanity-checking before a destructive change.

`./rotate.sh --skip-railway` rotates the server without touching the
Railway env var — useful when you want to poke at the new server
manually before flipping traffic to it.

## State

Terraform state is local (`terraform.tfstate`). For solo use this is
fine — back up the state file with your normal backup process. For
team use, switch to a remote backend (S3-compatible, Terraform Cloud,
or Hetzner's own object storage).

**Never commit `terraform.tfvars`** — it holds `hcloud_token` and
`proxy_secret`. Already listed in `.gitignore`.

## Cost

At current pricing (2026-07):
- **cpx11** (recommended): €4.35/mo ≈ AUD $7.20/mo
- Bandwidth included: 20TB/mo (we use ~1GB/mo)
- Hourly billing — destroy + recreate freely during testing

## Migrating from the DO module

The two modules coexist. Recommended flow:

1. Deploy Hetzner: `./rotate.sh --yes` in this directory.
2. Verify Railway is now pointing at the Hetzner URL and results are
   flowing (`/api/admin/probe-ra-results` returns 200).
3. Leave DO running for 24-48h as fallback (~$0.30/day).
4. Once stable, `terraform destroy` in `infra/ra-proxy/` to stop
   paying for the blocked DO droplet.
