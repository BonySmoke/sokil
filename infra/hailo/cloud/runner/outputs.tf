# Read by infra/hailo/hailo.sh to follow, fetch and destroy the run.

output "run_id" {
  value = var.run_id
}

output "suite_key" {
  value = var.suite_key
}

output "instance_id" {
  value = one(flatten(aws_ec2_fleet.runner.fleet_instance_set[*].instance_ids))
}
