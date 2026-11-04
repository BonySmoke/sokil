# What outlives a compile run: the bucket holding the Hailo AI Software Suite between
# runs, and the role, instance profile and security group each run's VM uses.
# Applied once; the runner stack reads its outputs from the local state file.

terraform {
  required_version = ">= 1.13"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.70"
    }
  }
}
