locals {
  persistent = data.terraform_remote_state.persistent.outputs

  tags = {
    Project   = "sokil"
    Component = "hailo-compiler"
    Name      = "sokil-hailo-${var.run_id}"
  }
}
