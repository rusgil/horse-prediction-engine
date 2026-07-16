output "droplet_id" {
  description = "DigitalOcean droplet ID — pass to doctl commands"
  value       = digitalocean_droplet.ra_proxy.id
}

output "droplet_ip" {
  description = "Droplet's public IPv4 address"
  value       = digitalocean_droplet.ra_proxy.ipv4_address
}

output "droplet_region" {
  description = "DigitalOcean region slug"
  value       = digitalocean_droplet.ra_proxy.region
}

output "sslip_hostname" {
  description = "sslip.io hostname derived from the droplet IP — Caddy serves LetsEncrypt-issued TLS on this name"
  value       = "${replace(digitalocean_droplet.ra_proxy.ipv4_address, ".", "-")}.sslip.io"
}

output "proxy_url" {
  description = "Full HTTPS URL to set as RA_PROXY_URL on Railway"
  value       = "https://${replace(digitalocean_droplet.ra_proxy.ipv4_address, ".", "-")}.sslip.io"
}

output "next_steps" {
  description = "What to do after apply completes"
  value = <<-EOT

    Terraform apply complete. Post-apply checklist:

    1. Update Railway env var:
         railway variables set RA_PROXY_URL=https://${replace(digitalocean_droplet.ra_proxy.ipv4_address, ".", "-")}.sslip.io

    2. Railway auto-redeploys (~90s). Watch:
         railway logs --deployment

    3. Once redeployed, test results seeding:
         curl -X POST -H 'x-cron-secret: <SECRET>' \\
           https://web-production-dec62.up.railway.app/api/admin/seed-ra-results/$(date +%Y-%m-%d)

    4. Verify /health from Railway's perspective:
         curl https://web-production-dec62.up.railway.app/api/health

    5. Wait 24-48h for stability confirmation. Then, if there's an OLD
       droplet still running from a manual rotation, destroy it:
         doctl compute droplet list --tag-name ra-proxy
         doctl compute droplet delete <OLD_ID>

    Droplet: ${digitalocean_droplet.ra_proxy.name}
    IP:      ${digitalocean_droplet.ra_proxy.ipv4_address}
    Region:  ${digitalocean_droplet.ra_proxy.region}
  EOT
}
