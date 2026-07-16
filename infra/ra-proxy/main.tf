# ------------------------------------------------------------------
# ra-proxy droplet — DigitalOcean-hosted reverse proxy that fronts
# Racing Australia for the horse prediction backend.
#
# All state is Terraform-managed:
#   - droplet + tag
#   - firewall rules
#   - bootstrap (Caddy, FastAPI proxy, systemd, self-heal) via
#     cloud-init user_data
#   - post-boot hostname fix (Caddy needs the droplet's actual IP)
#
# To rotate to a new IP (WAF-block recovery):
#   terraform apply -replace=digitalocean_droplet.ra_proxy
# or bump var.rotation_id and:
#   terraform apply
# ------------------------------------------------------------------

locals {
  # Read the proxy.py source directly from the sibling droplet-proxy
  # directory. Whatever's committed there is what gets deployed.
  proxy_py_content = file("${path.module}/../../droplet-proxy/proxy.py")
}

# Provision a fresh Ubuntu 24.04 droplet with cloud-init bootstrapping
# the entire proxy stack. New droplet ⇒ new outbound IP, which is the
# whole point when RA WAF-blocks us.
#
# Chicken-and-egg: the Caddyfile needs the droplet's own IP baked in
# so LetsEncrypt can issue a cert. Terraform can't reference a
# resource's own attributes in its user_data, so we bake in a
# placeholder token here and let the null_resource below rewrite it
# once the droplet has an IP.
resource "digitalocean_droplet" "ra_proxy" {
  name     = "ra-proxy-${var.rotation_id}"
  image    = "ubuntu-24-04-x64"
  region   = var.region
  size     = var.droplet_size
  ssh_keys = var.ssh_key_fingerprints
  tags     = ["ra-proxy"]

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
# host. Runs on every apply where the droplet's IP changes.
resource "null_resource" "caddy_hostname_fix" {
  triggers = {
    droplet_id = digitalocean_droplet.ra_proxy.id
    droplet_ip = digitalocean_droplet.ra_proxy.ipv4_address
  }

  connection {
    type    = "ssh"
    host    = digitalocean_droplet.ra_proxy.ipv4_address
    user    = "root"
    timeout = "5m"
    agent   = true
  }

  provisioner "remote-exec" {
    inline = [
      # Wait for cloud-init to install packages and write files
      "cloud-init status --wait || true",
      # Substitute the real hostname derived from the droplet IP
      "REAL_HOST=$(echo ${digitalocean_droplet.ra_proxy.ipv4_address} | tr '.' '-').sslip.io",
      "sed -i \"s/PLACEHOLDER_HOSTNAME.sslip.io/$${REAL_HOST}/g\" /etc/caddy/Caddyfile",
      "systemctl reload caddy || systemctl restart caddy",
      # LetsEncrypt cert issue takes ~30-60s on first request
      "for i in $(seq 1 12); do",
      "  sleep 10",
      "  CODE=$(curl -sk -o /dev/null -w '%%{http_code}' https://localhost/health || echo 000)",
      "  echo \"health probe $i: $CODE\"",
      "  if [ \"$CODE\" = \"200\" ]; then break; fi",
      "done"
    ]
  }

  depends_on = [digitalocean_droplet.ra_proxy]
}

# Firewall: SSH (rate-limited by ufw), HTTP + HTTPS, ICMP.
# Applied to the droplet by tag so a rotation carries the rules over.
resource "digitalocean_firewall" "ra_proxy" {
  name        = "ra-proxy-${var.rotation_id}"
  droplet_ids = [digitalocean_droplet.ra_proxy.id]

  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = [var.backend_source_ip]
  }

  inbound_rule {
    protocol         = "icmp"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}
