
output "s3_bucket_name" {
  value = aws_s3_bucket.inference_data.bucket
}

output "batch_job_queue_arn" {
  value = aws_batch_job_queue.gpu_job_queue.arn
}
