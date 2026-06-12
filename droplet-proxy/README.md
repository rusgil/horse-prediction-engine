# Racing Australia proxy — droplet setup

Tiny FastAPI app that forwards requests to `racingaustralia.horse` from a DigitalOcean droplet. The horse-prediction backend (running on Railway) points its `RacingAustraliaClient` at this proxy URL when set; from RA's perspective the requests come from the droplet's IP, not Railway's WAF-blocked one.

## What you need before you start

1. A DigitalOcean droplet (Ubuntu 24.04, $6/mo Basic, Sydney region preferred). See main project notes for product picks.
2. A domain you control with DNS access (so we can use `ra-proxy.yourdomain.com` with auto-TLS).
3. The droplet's public IPv4 address.
4. The SSH private key matching the public key you uploaded when creating the droplet.

## Step 1 — point a subdomain at the droplet

In your DNS provider, add an A record:

```
ra-proxy.yourdomain.com  A  <DROPLET-PUBLIC-IP>
```

Wait 1–2 minutes for propagation. Verify with:

```
dig +short ra-proxy.yourdomain.com
```

Should return the droplet IP.

## Step 2 — SSH into the droplet

```
ssh root@<DROPLET-PUBLIC-IP>
```

## Step 3 — install packages

```
apt update && apt upgrade -y
apt install -y python3-venv python3-pip caddy ufw
```

Caddy installs as `/usr/bin/caddy` and creates the service file automatically.

## Step 4 — create the service user + working dir

```
useradd -r -s /bin/false ra-proxy
mkdir -p /opt/ra-proxy
chown ra-proxy:ra-proxy /opt/ra-proxy
```

## Step 5 — copy proxy code onto the droplet

From your laptop (or use `scp`):

```
scp droplet-proxy/proxy.py droplet-proxy/requirements.txt root@<DROPLET-PUBLIC-IP>:/opt/ra-proxy/
```

Back on the droplet:

```
chown -R ra-proxy:ra-proxy /opt/ra-proxy
```

## Step 6 — install Python dependencies in a venv

```
sudo -u ra-proxy bash -c "
  cd /opt/ra-proxy
  python3 -m venv venv
  ./venv/bin/pip install -r requirements.txt
"
```

## Step 7 — generate a strong proxy secret

```
PROXY_SECRET=$(openssl rand -hex 32)
echo "PROXY_SECRET=$PROXY_SECRET" > /etc/ra-proxy.env
chmod 600 /etc/ra-proxy.env
chown root:root /etc/ra-proxy.env
echo "Save this — you'll add it to Railway env vars: $PROXY_SECRET"
```

## Step 8 — install the systemd unit

```
cp droplet-proxy/ra-proxy.service /etc/systemd/system/ra-proxy.service
systemctl daemon-reload
systemctl enable --now ra-proxy
systemctl status ra-proxy   # should show 'active (running)'
```

If it fails, check logs:

```
journalctl -u ra-proxy -n 50
```

Quick local test (still on the droplet):

```
curl -s http://127.0.0.1:8000/health
# should return {"status":"ok"}
```

## Step 9 — install Caddy config

Replace `ra-proxy.YOURDOMAIN.com` in `Caddyfile` with your actual subdomain, then:

```
cp droplet-proxy/Caddyfile /etc/caddy/Caddyfile
# edit the domain placeholder before reloading:
sed -i "s/ra-proxy.YOURDOMAIN.com/ra-proxy.example.com/" /etc/caddy/Caddyfile
systemctl reload caddy
```

Caddy will provision Let's Encrypt TLS automatically. Watch logs:

```
journalctl -u caddy -f
# wait for the 'certificate obtained successfully' line, then Ctrl+C
```

## Step 10 — lock down firewall

```
ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw --force enable
ufw status
```

This blocks port 8000 from the outside — uvicorn only listens on 127.0.0.1 anyway, so Caddy is the only thing that can reach it.

## Step 11 — verify externally

From your laptop:

```
curl -s https://ra-proxy.yourdomain.com/health
# should return {"status":"ok"}

# Without the secret — expect 403:
curl -s -w "\n%{http_code}\n" "https://ra-proxy.yourdomain.com/proxy/FreeFields/Calendar.aspx?State=NSW"

# With the secret — expect HTML (RA Calendar page):
curl -s -H "X-Proxy-Secret: $PROXY_SECRET" "https://ra-proxy.yourdomain.com/proxy/FreeFields/Calendar.aspx?State=NSW" | head -20
```

If the last command returns HTML containing `<html` and meeting names, **you're done.** RA is reachable from this droplet's IP.

## Step 12 — wire Railway to use the proxy

Two env vars on the Railway service:

```
RA_PROXY_URL=https://ra-proxy.yourdomain.com
RA_PROXY_SECRET=<the secret from Step 7>
```

(I'll ship the matching code change in `horse_engine/clients/racing_australia.py` once you have the droplet running — it'll detect those vars and route through the proxy automatically.)

## Ongoing

- Caddy auto-renews TLS — no action needed.
- DigitalOcean automatic security updates can be enabled in dashboard: Settings → Automatic OS updates.
- If RA ever WAF-blocks the droplet too:
  1. Snapshot the droplet (~10s)
  2. Destroy it
  3. Recreate from snapshot in the same region
  4. New IPv4 assigned
  5. Update DNS A record
  6. Wait 1–2 min for propagation
  Total downtime: ~5 minutes for the entire reset, no code changes.

## File map

```
droplet-proxy/
├── proxy.py             # FastAPI app — forwards GETs to RA
├── requirements.txt     # fastapi + uvicorn + httpx
├── ra-proxy.service     # systemd unit
├── Caddyfile            # TLS + reverse proxy config
└── README.md            # this file
```
