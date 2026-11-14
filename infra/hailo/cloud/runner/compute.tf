# The VM: a launch template carrying bootstrap.sh, and the fleet that starts one
# spot instance from it.

resource "aws_launch_template" "runner" {
  name_prefix            = "sokil-hailo-"
  image_id               = data.aws_ssm_parameter.ami.insecure_value
  instance_type          = var.instance_type
  vpc_security_group_ids = [local.persistent.security_group_id]

  iam_instance_profile {
    name = local.persistent.instance_profile
  }

  # Powering off deletes the instance, which is how the run ends: on success,
  # on failure and when the kill switch in bootstrap.sh fires.
  instance_initiated_shutdown_behavior = "terminate"

  metadata_options {
    http_tokens = "required"
  }

  block_device_mappings {
    device_name = data.aws_ami.dlami.root_device_name

    ebs {
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  # The settings as shell assignments, then the script itself, so the script
  # needs no template escaping and shellcheck can read it as it is.
  user_data = base64encode(join("\n", [
    "#!/bin/bash",
    "export HAILO_BUCKET='${local.persistent.bucket}'",
    "export HAILO_RUN_ID='${var.run_id}'",
    "export HAILO_SUITE_KEY='${var.suite_key}'",
    "export HAILO_MAX_MINUTES='${var.max_minutes}'",
    "export AWS_DEFAULT_REGION='${local.persistent.region}'",
    file("${path.module}/bootstrap.sh"),
  ]))

  # The provider's default_tags reach the template but not what a fleet
  # launches from it.
  tag_specifications {
    resource_type = "instance"
    tags          = local.tags
  }

  tag_specifications {
    resource_type = "volume"
    tags          = local.tags
  }
}

# An instant fleet rather than a plain instance: given a subnet in every zone,
# AWS launches in one that has spot capacity, in a single call. A plain
# instance is pinned to one zone, and the provider retries a zone without
# capacity for over a quarter of an hour before giving up.
#
# Without capacity anywhere it launches nothing; the postcondition turns that
# into a failed apply. It never falls back to the on-demand price.
resource "aws_ec2_fleet" "runner" {
  type                = "instant"
  terminate_instances = true

  launch_template_config {
    launch_template_specification {
      launch_template_id = aws_launch_template.runner.id
      version            = aws_launch_template.runner.latest_version
    }

    dynamic "override" {
      for_each = split(" ", local.persistent.subnet_ids)

      content {
        subnet_id = override.value
      }
    }
  }

  target_capacity_specification {
    default_target_capacity_type = "spot"
    total_target_capacity        = 1
  }

  # All or nothing: one instance or none. The provider sends a minimum target
  # capacity regardless, which AWS only accepts on a fleet restricted to a
  # single instance type or zone; one type is all the quota fits anyway, and
  # it leaves the fleet free to choose the zone.
  spot_options {
    allocation_strategy  = "price-capacity-optimized"
    single_instance_type = true
    min_target_capacity  = 1
  }

  lifecycle {
    postcondition {
      condition     = length(flatten(self.fleet_instance_set[*].instance_ids)) > 0
      error_message = "No ${var.instance_type} spot capacity in any zone of ${local.persistent.region}; try again later."
    }
  }
}
