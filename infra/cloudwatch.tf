resource "aws_cloudwatch_log_group" "batch_logs" {
  name              = "/aws/batch/${var.project_name}"
  retention_in_days = 7

  tags = {
    Name        = "${var.project_name}-batch-logs"
    Environment = "production"
  }
}