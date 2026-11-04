provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "sokil"
      Component = "hailo-compiler"
    }
  }
}
