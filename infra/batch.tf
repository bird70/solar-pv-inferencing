
resource "aws_batch_compute_environment" "gpu_compute_env" {
  compute_environment_name = "${var.project_name}-gpu-compute-env"
  type                     = "MANAGED"
  service_role             = aws_iam_role.batch_service_role.arn

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    min_vcpus           = 0
    max_vcpus           = 16
    desired_vcpus       = 0
    instance_type       = ["g6.xlarge"]
    subnets             = data.aws_subnets.default.ids
    security_group_ids  = [data.aws_security_group.default.id]
    ec2_key_pair        = var.key_pair_name
    instance_role       = aws_iam_instance_profile.ecs_instance_profile.arn
  }
}

resource "aws_batch_job_queue" "gpu_job_queue" {
  name     = "${var.project_name}-gpu-job-queue"
  state    = "ENABLED"
  priority = 1
  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu_compute_env.arn
  }
}

resource "aws_batch_job_definition" "gpu_job_def" {
  name                  = "${var.project_name}-gpu-job-definition"
  type                  = "container"
  platform_capabilities = ["EC2"]

  container_properties = jsonencode({
    image  = var.ecr_image_url
    vcpus  = 4
    memory = 8192
    resourceRequirements = [
      {
        type  = "GPU"
        value = "1"
      }
    ]
    environment = [
      {
        name  = "S3_BUCKET"
        value = var.s3_bucket_name
      },
      {
        name  = "ROBOFLOW_MODEL_ID"
        value = var.roboflow_model_id
      },
      {
        name  = "ROBOFLOW_API_KEY"
        value = var.roboflow_api_key
      }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch_logs.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "gpu-inference"
      }
    }
  })
}
