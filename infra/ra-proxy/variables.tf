variable "do_token" {
  description = "DigitalOcean API token (read+write). Get one at https://cloud.digitalocean.com/account/api/tokens"
  type        = string
  sensitive   = true
}

variable "proxy_secret" {
  description = "Shared secret between Railway backend and this proxy. Set the SAME value as the RA_PROXY_SECRET env var on Railway."
  type        = string
  sensitive   = true
}

variable "ssh_key_fingerprints" {
  description = "SSH key fingerprint(s) already registered with DigitalOcean. Get with: doctl compute ssh-key list --format FingerPrint --no-header"
  type        = list(string)
}

variable "region" {
  description = "DigitalOcean region slug"
  type        = string
  default     = "syd1" # Sydney — closest to Racing Australia + Railway (both AU-hosted)
}

variable "droplet_size" {
  description = "Droplet size slug. s-1vcpu-512mb-10gb is the cheapest that runs Caddy + FastAPI comfortably."
  type        = string
  default     = "s-1vcpu-1gb"
}

variable "rotation_id" {
  description = "Bump this value (e.g. YYYYMMDD-N) to force a fresh droplet with a new IP. Use when RA WAF-blocks the current IP."
  type        = string
  default     = "1"
}

variable "backend_source_ip" {
  description = "Optional — restrict inbound 443 to Railway's egress CIDR when known. Leave as 0.0.0.0/0 for public access."
  type        = string
  default     = "0.0.0.0/0"
}
