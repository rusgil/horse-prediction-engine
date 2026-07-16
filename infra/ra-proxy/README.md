# ra-proxy · Terraform infrastructure

Terraform-managed DigitalOcean droplet that fronts Racing Australia for the horse prediction backend. Replaces the manual `scripts/migrate-ra-proxy.sh` playbook with declarative infrastructure.

## What this manages

| Resource | Purpose |
|---|---|
| `digitalocean_droplet.ra_proxy` | Ubuntu 24.04 droplet running Caddy + FastAPI proxy |
| `digitalocean_firewall.ra_proxy` | SSH / HTTP / HTTPS / ICMP inbound; all outbound allowed |
| `null_resource.caddy_hostname_fix` | Post-boot SSH to rewrite Caddyfile hostname to the droplet's actual IP |
| `cloud-init.yaml.tpl` | Full bootstrap — packages, users, systemd services, self-heal timer |

The `droplet-proxy/proxy.py` file is read at plan time via `file()` so whatever's committed on the branch you apply from is what gets deployed. No SCP step, no manual copy.

## One-time setup

**1. Install tools**
```bash
brew install terraform doctl
doctl auth init                    # paste a DigitalOcean API token
```

**2. Register your SSH key with DigitalOcean**
```bash
doctl compute ssh-key create my-laptop --public-key-file ~/.ssh/id_ed25519.pub
doctl compute ssh-key list --format FingerPrint,Name
```
Copy the fingerprint(s) — you'll paste them into `terraform.tfvars`.

**3. Create your local vars file**
```bash
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars
```
Fill in `do_token`, `proxy_secret`, `ssh_key_fingerprints`. `terraform.tfvars` is git-ignored — it never gets committed.

**4. Initialise Terraform**
```bash
terraform init
```

## First provision (new droplet, no existing state)

If this is the first Terraform-managed droplet — you're starting fresh:

```bash
terraform plan            # sanity check
terraform apply           # create droplet + firewall + bootstrap
```

Takes ~3-5 min. On completion Terraform prints a `next_steps` output with the exact `RA_PROXY_URL` value to set on Railway.

## Importing an existing hand-rolled droplet

If you already have a droplet running (the `170.64.147.74` one), import it into state before your first `apply`:

```bash
# Find the droplet ID
doctl compute droplet list --tag-name ra-proxy --format ID,Name

# Import into state
terraform import digitalocean_droplet.ra_proxy <DROPLET_ID>

# Compare state to config
terraform plan
```

The plan will show diffs where the imported droplet doesn't match this config (e.g. it wasn't provisioned via `templatefile`). Decide case-by-case whether to accept the drift or `terraform apply -replace` to rebuild cleanly.

## Rotating to a new IP (the common case)

RA WAF-blocked your outbound IP again. Two options:

**Option A — bump rotation_id (declarative, preferred)**
```bash
# Edit terraform.tfvars, change rotation_id to today's date, then:
terraform apply
```
Because `rotation_id` is baked into the droplet name and `create_before_destroy = true`, Terraform provisions the new droplet FIRST, waits for `/health` to return 200, and only then destroys the old one. Zero downtime.

**Option B — direct replace**
```bash
terraform apply -replace=digitalocean_droplet.ra_proxy
```
Same behaviour, different UX. Doesn't require editing `terraform.tfvars`.

## After every apply — update Railway

The Terraform output includes the exact command:

```
railway variables set RA_PROXY_URL=https://<NEW_IP_AS_SLIP_HOST>
```

Railway auto-redeploys on env var change (~90s). Then verify:

```bash
curl -X POST -H 'x-cron-secret: <SECRET>' \
  https://web-production-dec62.up.railway.app/api/admin/seed-ra-results/$(date +%Y-%m-%d)
```

If results seed cleanly, you're done. Wait 24-48h before destroying any older droplets that may still be running from prior manual rotations.

## Rotating the proxy secret

The `PROXY_SECRET` shared with Railway lives in `terraform.tfvars`. To rotate:

1. Generate a fresh value: `openssl rand -hex 32`
2. Update `terraform.tfvars`
3. `terraform apply` — this changes cloud-init user_data, which forces droplet recreation with the new secret baked in
4. Update the Railway `RA_PROXY_SECRET` env var to match
5. Railway redeploys → both sides use the new secret

## State backend

Currently local (state file on your laptop). For team use, uncomment the S3-compatible backend block in `versions.tf` and configure a DigitalOcean Spaces bucket:

```hcl
terraform {
  backend "s3" {
    endpoint                    = "https://syd1.digitaloceanspaces.com"
    region                      = "us-east-1"    # required but ignored by DO
    key                         = "ra-proxy/terraform.tfstate"
    bucket                      = "your-tfstate-bucket"
    skip_credentials_validation = true
    skip_metadata_api_check     = true
    skip_region_validation      = true
  }
}
```

Then `terraform init -reconfigure` to migrate the state.

## Troubleshooting

**`caddy_hostname_fix` fails with SSH timeout**
Cloud-init hasn't finished. Increase the connection timeout in `main.tf` or re-run `terraform apply` (it's safe — the fix is idempotent).

**Droplet up, but /health returns 502 or 404**
`ra-proxy` service didn't start. SSH in and check:
```bash
ssh root@<IP>
systemctl status ra-proxy
tail -50 /var/log/ra-proxy-boot.log
```

**Certificate errors on first request**
LetsEncrypt takes 30-60s to issue on first request. `terraform apply` polls up to 2 min; if it still fails, Caddy logs will show the exact error:
```bash
journalctl -u caddy -n 200
```

**RA still returns 403 after rotation**
The new IP is also blocked. Options:
- Change region (`syd1` → `sfo3` for a US-based IP, then rotate back later)
- Wait 24h for RA to clear the block on the previous IP first
- Contact RA to request whitelisting

## Cost

- Droplet `s-1vcpu-1gb`: **$6/mo** (Sydney)
- Snapshot storage (during rotation): **~$0.05/GB/mo**, negligible
- Bandwidth: within the droplet's included 1TB/mo

Total: ~$6/mo — same as manual setup.
