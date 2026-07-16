terraform {
  required_version = ">= 1.5.0"

  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.44"
    }
  }

  # Local state for solo use. For team use, switch to DO Spaces
  # (S3-compatible) or Terraform Cloud — see README.
}

provider "digitalocean" {
  token = var.do_token
}
