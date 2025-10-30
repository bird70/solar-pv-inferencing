
resource "aws_s3_bucket" "inference_data" {
  bucket = var.s3_bucket_name

  tags = {
    Name        = var.s3_bucket_name
    Environment = "production"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "inference_data_lifecycle" {
  bucket = aws_s3_bucket.inference_data.id

  rule {
    id     = "delete_after_30_days"
    status = "Enabled"

    filter {}

    expiration {
      days = 30
    }
  }
}
