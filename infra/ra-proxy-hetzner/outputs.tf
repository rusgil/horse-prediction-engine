output "server_id" {
  description = "Hetzner Cloud server ID"
  value       = hcloud_server.ra_proxy.id
}

# Kept as droplet_ip so rotate.sh, which was written against the DO
# module, works against this module without renaming. If we ever
# retire the DO module we can rename this too.
output "droplet_ip" {
  description = "Server's public IPv4 address (named droplet_ip for rotate.sh compat)"
  value       = hcloud_server.ra_proxy.ipv4_address
}

output "server_location" {
  description = "Hetzner Cloud location code"
  value       = hcloud_server.ra_proxy.location
}

output "sslip_hostname" {
  description = "sslip.io hostname derived from the server IP — Caddy serves LetsEncrypt-issued TLS on this name"
  value       = "${replace(hcloud_server.ra_proxy.ipv4_address, ".", "-")}.sslip.io"
}

output "proxy_url" {
  description = "Full HTTPS URL to set as RA_PROXY_URL on Railway"
  value       = "https://${replace(hcloud_server.ra_proxy.ipv4_address, ".", "-")}.sslip.io"
}

output "next_steps" {
  description = "What to do after apply completes"
  value = <<-EOT

    Terraform apply complete (Hetzner Cloud). Post-apply checklist:

    1. Update Railway env var:
         railway variables set RA_PROXY_URL=https://${replace(hcloud_server.ra_proxy.ipv4_address, ".", "-")}.sslip.io

    2. Railway auto-redeploys (~90s). Watch:
         railway logs --deployment

    3. Once redeployed, test results seeding:
         curl -X POST -H 'x-cron-secret: <SECRET>' \\
           https://web-production-dec62.up.railway.app/api/admin/seed-ra-results/$(date +%Y-%m-%d)

    4. Verify /health from Railway's perspective:
         curl https://web-production-dec62.up.railway.app/api/health

    5. Wait 24-48h for stability confirmation. If it holds, tear down
       the old DigitalOcean droplet-based module state — it's paying
       ~$6/mo to sit there blocked.

    Server:   ${hcloud_server.ra_proxy.name}
    IP:       ${hcloud_server.ra_proxy.ipv4_address}
    Location: ${hcloud_server.ra_proxy.location}
    Type:     ${hcloud_server.ra_proxy.server_type}
  EOT
}
