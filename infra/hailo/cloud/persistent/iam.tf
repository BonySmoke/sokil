# The VM's identity. Its policies are in data.tf.

resource "aws_iam_role" "runner" {
  name_prefix        = "sokil-hailo-runner-"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

# Session Manager, for a shell on the VM without an open port or a key pair.
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.runner.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "bucket" {
  name   = "bucket"
  role   = aws_iam_role.runner.id
  policy = data.aws_iam_policy_document.bucket.json
}

resource "aws_iam_instance_profile" "runner" {
  name_prefix = "sokil-hailo-runner-"
  role        = aws_iam_role.runner.name
}
