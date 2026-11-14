data "terraform_remote_state" "persistent" {
  backend = "local"

  config = {
    path = "${path.module}/../persistent/terraform.tfstate"
  }
}

# The NVIDIA driver, Docker and the NVIDIA container toolkit come preinstalled,
# so a run spends no time on them. The suite brings its own CUDA; the host
# only needs a driver new enough for it.
data "aws_ssm_parameter" "ami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
}

# The root device's name, which the launch template needs to set the volume up.
data "aws_ami" "dlami" {
  owners = ["amazon"]

  filter {
    name   = "image-id"
    values = [data.aws_ssm_parameter.ami.insecure_value]
  }
}
