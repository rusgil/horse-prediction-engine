# ------------------------------------------------------------------
# ra-proxy server (Hetzner Cloud port) — reverse proxy for Racing
# Australia, deployed on a different-ASN VPS to escape the DO IP
# reputation block that hit us on 2026-07-17.
#
# Same design as the sibling ra-proxy (DigitalOcean) module:
#   - Server + tags
#   - Firewall (via hcloud_firewall + firewall_ids on the server)
#   - Bootstrap via cloud-init user_data (Caddy, curl_cffi-based
#     FastAPI proxy, systemd, self-heal)
#   - Post-boot Caddy hostname fix once the server has an IP
#
# To rotate to a new IP (WAF-block recovery):
#   bump var.rotation_id in terraform.tfvars
#   ./rotate.sh --yes
# ------------------------------------------------------------------

locals {
  # Shared source-of-truth for the proxy code, read from the sibling
  # droplet-proxy/ directory (kept even after DO -> Hetzner move
  # because the code is provider-agnostic).
  proxy_py_content = file("${path.module}/../../droplet-proxy/proxy.py")
}

# Resolve SSH key names → IDs so we can attach them at server creation.
# Hetzner's provider takes a list of IDs on the server resource.
data "hcloud_ssh_key" "keys" {
  for_each = toset(var.ssh_key_names)
  name     = each.value
}

# Ubuntu 24.04 image on Hetzner. Same LTS as the DO module used.
data "hcloud_image" "ubuntu" {
  name              = "ubuntu-24.04"
  with_architecture = startswith(var.server_type, "cax") ? "arm" : "x86"
}

resource "hcloud_firewall" "ra_proxy" {
  name = "ra-proxy-${var.rotation_id}"

  # SSH — open, rate-limited at the OS level by fail2ban/ufw
  # (kept lightweight because the whole box is disposable).
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  # HTTP — Caddy uses this for LetsEncrypt HTTP-01 challenge only,
  # then redirects everything to HTTPS.
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "80"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  # HTTPS — the actual proxy. Optionally scoped down to Railway's
  # egress CIDR via var.backend_source_ip if known.
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "443"
    source_ips = [var.backend_source_ip]
  }

  # ICMP — leave open so ping / traceroute work when we're diagnosing.
  rule {
    direction  = "in"
    protocol   = "icmp"
    source_ips = ["0.0.0.0/0", "::/0"]
  }
}

# Provision a fresh Ubuntu 24.04 server with cloud-init bootstrapping
# the entire proxy stack. New server ⇒ new outbound IP, which is the
# whole point when RA WAF-blocks us.
#
# Chicken-and-egg: the Caddyfile needs the server's own IP baked in
# so LetsEncrypt can issue a cert. Terraform can't reference a
# resource's own attributes in its user_data, so we bake in a
# placeholder token here and let the null_resource below rewrite it
# once the server has an IP.
resource "hcloud_server" "ra_proxy" {
  name         = "ra-proxy-${var.rotation_id}"
  image        = data.hcloud_image.ubuntu.id
  server_type  = var.server_type
  location     = var.location
  ssh_keys     = [for k in data.hcloud_ssh_key.keys : k.id]
  firewall_ids = [hcloud_firewall.ra_proxy.id]
  labels = {
    role = "ra-proxy"
  }

  user_data = templatefile("${path.module}/cloud-init.yaml.tpl", {
    proxy_secret     = var.proxy_secret
    proxy_hostname   = "PLACEHOLDER_HOSTNAME.sslip.io"
    proxy_py_content = local.proxy_py_content
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Post-apply: SSH in, rewrite the Caddyfile hostname to the real IP,
# reload Caddy so LetsEncrypt issues a cert for the actual sslip.io
# host. Runs on every apply where the server's IP changes.
resource "null_resource" "caddy_hostname_fix" {
  triggers = {
    server_id = hcloud_server.ra_proxy.id
    server_ip = hcloud_server.ra_proxy.ipv4_address
  }

  connection {
    type    = "ssh"
    host    = hcloud_server.ra_proxy.ipv4_address
    user    = "root"
    timeout = "5m"
    agent   = true
  }

  provisioner "remote-exec" {
    inline = [
      # Wait for cloud-init to install packages and write files
      "cloud-init status --wait || true",
      # Substitute the real hostname derived from the server IP
      "REAL_HOST=$(echo ${hcloud_server.ra_proxy.ipv4_address} | tr '.' '-').sslip.io",
      "sed -i \"s/PLACEHOLDER_HOSTNAME.sslip.io/$${REAL_HOST}/g\" /etc/caddy/Caddyfile",
      "grep -q \"$${REAL_HOST}\" /etc/caddy/Caddyfile || { echo 'hostname substitution failed' >&2; exit 1; }",
      "systemctl reload caddy || systemctl restart caddy",
      # Caddy `tls internal` mints a self-signed cert instantly (no ACME wait,
      # no Let's Encrypt rate limits). Probe the real sslip hostname resolved to
      # localhost so it matches the site block + self-signed cert (-k trusts it).
      "for i in $(seq 1 12); do",
      "  sleep 5",
      "  CODE=$(curl -sk -o /dev/null -w '%%{http_code}' --resolve $${REAL_HOST}:443:127.0.0.1 https://$${REAL_HOST}/health || echo 000)",
      "  echo \"health probe $i: $CODE\"",
      "  if [ \"$CODE\" = \"200\" ]; then break; fi",
      "done"
    ]
  }

  depends_on = [hcloud_server.ra_proxy]
}
