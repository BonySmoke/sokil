# Read by the runner stack and by infra/hailo/hailo.sh.

output "region" {
  value = var.region
}

output "bucket" {
  value = aws_s3_bucket.hailo.id
}

output "instance_profile" {
  value = aws_iam_instance_profile.runner.name
}

output "security_group_id" {
  value = aws_security_group.runner.id
}

# Space-separated rather than a list, so hailo.sh can read it with `output
# -raw` and loop over it without a JSON parser.
output "subnet_ids" {
  value = join(" ", [for subnet in aws_default_subnet.runner : subnet.id])
}
