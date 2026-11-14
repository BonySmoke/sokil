provider "aws" {
  region = local.persistent.region

  default_tags {
    tags = {
      Project   = "sokil"
      Component = "hailo-compiler"
    }
  }
}
