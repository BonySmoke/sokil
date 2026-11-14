# One compile run's VM: a spot GPU instance that runs bootstrap.sh from user
# data, writes its results to the bucket and terminates itself. Applied and
# destroyed per run by infra/hailo/hailo.sh; see docs/accelerators.md.

terraform {
  required_version = ">= 1.13"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.70"
    }
  }
}
