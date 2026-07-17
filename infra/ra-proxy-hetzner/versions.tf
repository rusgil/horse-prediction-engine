terraform {
  required_version = ">= 1.5.0"

  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "~> 1.48"
    }
  }

  # Local state for solo use. For team use, switch to a remote backend
  # (S3-compatible object storage or Terraform Cloud). Same guidance
  # as the sibling ra-proxy (DigitalOcean) module.
}

provider "hcloud" {
  token = var.hcloud_token
}
