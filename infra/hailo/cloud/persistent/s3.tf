# suite/ holds the Hailo data flow compiler docker image as the tarball and is the only thing kept;
# runs/<id>/ holds a run's code, inputs and results, and is deleted once they are downloaded.
resource "aws_s3_bucket" "hailo" {
  bucket_prefix = "sokil-hailo-"
}

resource "aws_s3_bucket_public_access_block" "hailo" {
  bucket = aws_s3_bucket.hailo.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "hailo" {
  bucket = aws_s3_bucket.hailo.id

  # An interrupted multipart upload of the 8.5 GB suite leaves parts behind that
  # are billed as storage but never listed.
  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }

  # A run's prefix is deleted when its results are fetched; this catches the
  # runs that never were.
  rule {
    id     = "expire-runs"
    status = "Enabled"

    filter {
      prefix = "runs/"
    }

    expiration {
      days = 7
    }
  }
}
