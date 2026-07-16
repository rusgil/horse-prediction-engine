#cloud-config
# ------------------------------------------------------------------
# ra-proxy droplet bootstrap
#
# Rendered from the Terraform template with:
#   - proxy_secret          shared secret with Railway backend
#   - proxy_hostname        sslip.io host for TLS (derived from IP)
#   - proxy_py_content      inlined proxy.py source
#
# This runs ONCE at first boot. Idempotent restarts are fine.
# ------------------------------------------------------------------

package_update: true
package_upgrade: true

packages:
  - python3-venv
  - python3-pip
  - ufw
  - debian-keyring
  - debian-archive-keyring
  - apt-transport-https

users:
  - name: ra-proxy
    system: true
    shell: /usr/sbin/nologin
    home: /opt/ra-proxy
    homedir: /opt/ra-proxy

write_files:
  # /etc/ra-proxy.env — read by the systemd service (EnvironmentFile).
  # 0600 root:root is intentional; only root and the ra-proxy user need read.
  - path: /etc/ra-proxy.env
    permissions: "0600"
    owner: root:root
    content: |
      PROXY_SECRET=${proxy_secret}

  # /opt/ra-proxy/proxy.py — the FastAPI proxy itself.
  - path: /opt/ra-proxy/proxy.py
    permissions: "0644"
    owner: ra-proxy:ra-proxy
    content: |
      ${indent(6, proxy_py_content)}

  # /opt/ra-proxy/requirements.txt — pinned deps
  - path: /opt/ra-proxy/requirements.txt
    permissions: "0644"
    owner: ra-proxy:ra-proxy
    content: |
      fastapi==0.115.5
      uvicorn[standard]==0.32.1
      httpx==0.27.2

  # /etc/systemd/system/ra-proxy.service — systemd unit
  - path: /etc/systemd/system/ra-proxy.service
    permissions: "0644"
    content: |
      [Unit]
      Description=Racing Australia proxy
      After=network.target

      [Service]
      Type=simple
      User=ra-proxy
      WorkingDirectory=/opt/ra-proxy
      EnvironmentFile=/etc/ra-proxy.env
      ExecStart=/opt/ra-proxy/venv/bin/uvicorn proxy:app --host 127.0.0.1 --port 8000
      Restart=on-failure
      RestartSec=5s

      NoNewPrivileges=true
      PrivateTmp=true
      ProtectSystem=strict
      ProtectHome=true
      ReadWritePaths=/opt/ra-proxy

      [Install]
      WantedBy=multi-user.target

  # Caddyfile — TLS-fronted reverse proxy to 127.0.0.1:8000.
  # Hostname is baked in from Terraform (${proxy_hostname}) so LetsEncrypt
  # can validate the sslip.io host on first boot.
  - path: /etc/caddy/Caddyfile
    permissions: "0644"
    content: |
      ${proxy_hostname} {
          reverse_proxy 127.0.0.1:8000
          @notproxy {
              not path /proxy/* /health
          }
          respond @notproxy 404
          tls {
              protocols tls1.2 tls1.3
          }
          encode gzip
          log {
              output file /var/log/caddy/access.log {
                  roll_size 10mb
                  roll_keep 5
              }
              format json
          }
      }

  # Health-check script — restarts services if /health fails.
  # Runs via systemd timer every 2 min.
  - path: /usr/local/bin/ra-proxy-healthcheck.sh
    permissions: "0755"
    content: |
      #!/bin/bash
      CODE=$$(curl -s -o /dev/null -w '%%{http_code}' -m 5 http://localhost:8000/ || echo 000)
      # Root returns 404 (no route) — that means FastAPI is UP.
      if [[ "$$CODE" != "200" && "$$CODE" != "404" ]]; then
          echo "$$(date -u -Iseconds) health failed (code=$$CODE) — restarting ra-proxy" >> /var/log/ra-proxy-healthcheck.log
          systemctl restart ra-proxy
      fi

  - path: /etc/systemd/system/ra-proxy-healthcheck.service
    permissions: "0644"
    content: |
      [Unit]
      Description=RA proxy self-heal
      [Service]
      Type=oneshot
      ExecStart=/usr/local/bin/ra-proxy-healthcheck.sh

  - path: /etc/systemd/system/ra-proxy-healthcheck.timer
    permissions: "0644"
    content: |
      [Unit]
      Description=RA proxy self-heal every 2 min
      [Timer]
      OnBootSec=2min
      OnUnitActiveSec=2min
      [Install]
      WantedBy=timers.target

runcmd:
  # Install Caddy from official Cloudsmith repo (idempotent on re-run)
  - |
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
    apt-get update
    apt-get install -y caddy

  # Python venv + deps
  - python3 -m venv /opt/ra-proxy/venv
  - /opt/ra-proxy/venv/bin/pip install --upgrade pip
  - /opt/ra-proxy/venv/bin/pip install -r /opt/ra-proxy/requirements.txt
  - chown -R ra-proxy:ra-proxy /opt/ra-proxy

  # UFW: SSH + HTTP + HTTPS only, everything else denied
  - ufw --force default deny incoming
  - ufw --force default allow outgoing
  - ufw allow 22/tcp
  - ufw allow 80/tcp
  - ufw allow 443/tcp
  - ufw --force enable

  # Enable + start services
  - systemctl daemon-reload
  - systemctl enable ra-proxy caddy ra-proxy-healthcheck.timer
  - systemctl start ra-proxy caddy ra-proxy-healthcheck.timer

  # Wait a moment then log status for post-boot diagnosis
  - sleep 5
  - systemctl status ra-proxy --no-pager > /var/log/ra-proxy-boot.log 2>&1 || true
  - systemctl status caddy --no-pager >> /var/log/ra-proxy-boot.log 2>&1 || true
