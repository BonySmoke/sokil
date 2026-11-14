# The default VPC can be left without its subnets, as this account's was, and
# EC2 then has nowhere to launch. This recreates the standard default subnet in
# each zone, where one is missing, and adopts it where it exists. One per zone,
# so the runner's fleet can launch in whichever zone has spot capacity.
#
# Destroying the stack leaves them in place; they cost nothing.
resource "aws_default_subnet" "runner" {
  for_each = toset(data.aws_availability_zones.available.names)

  availability_zone = each.value
}

# Nothing inbound: the VM only reaches out, to S3, Session Manager and PyPI.
resource "aws_security_group" "runner" {
  name_prefix = "sokil-hailo-runner-"
  description = "Hailo compile VM: outbound only"
  vpc_id      = data.aws_vpc.default.id
}

resource "aws_vpc_security_group_egress_rule" "ipv4" {
  security_group_id = aws_security_group.runner.id
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_vpc_security_group_egress_rule" "ipv6" {
  security_group_id = aws_security_group.runner.id
  ip_protocol       = "-1"
  cidr_ipv6         = "::/0"
}
