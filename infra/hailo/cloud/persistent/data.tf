data "aws_vpc" "default" {
  default = true
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_iam_policy_document" "assume_ec2" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

# Reads the suite and its run's inputs, writes its results. It cannot touch the
# suite, so a run gone wrong cannot cost the 8.5 GB upload.
data "aws_iam_policy_document" "bucket" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.hailo.arn]
  }

  statement {
    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.hailo.arn}/suite/*",
      "${aws_s3_bucket.hailo.arn}/runs/*",
    ]
  }

  statement {
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.hailo.arn}/runs/*"]
  }
}
