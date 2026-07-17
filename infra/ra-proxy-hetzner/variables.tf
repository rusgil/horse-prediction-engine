variable "hcloud_token" {
  description = "Hetzner Cloud API token (Read & Write). Generate at https://console.hetzner.cloud → your project → Security → API Tokens."
  type        = string
  sensitive   = true
}

variable "proxy_secret" {
  description = "Shared secret between Railway backend and this proxy. Set the SAME value as the RA_PROXY_SECRET env var on Railway."
  type        = string
  sensitive   = true
}

variable "ssh_key_names" {
  description = "SSH key name(s) already registered with Hetzner (via console or hcloud CLI). Get with: hcloud ssh-key list."
  type        = list(string)
}

variable "location" {
  description = "Hetzner Cloud location code. sin=Singapore (closest to RA), ash=Ashburn US, fsn1=Falkenstein DE."
  type        = string
  default     = "sin"
}

variable "server_type" {
  description = "Hetzner server type. cpx11 (AMD 2vCPU/2GB) is ideal for a proxy that uses <50MB RAM."
  type        = string
  default     = "cpx11"
}

variable "rotation_id" {
  description = "Bump this value (e.g. YYYYMMDD-N) to force a fresh server with a new IP. Use when RA WAF-blocks the current IP."
  type        = string
  default     = "1"
}

variable "backend_source_ip" {
  description = "Optional — restrict inbound 443 to Railway's egress CIDR when known. Leave as 0.0.0.0/0 for public access."
  type        = string
  default     = "0.0.0.0/0"
}
