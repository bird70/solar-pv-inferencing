
variable "project_name" {
  description = "solar-pv-panel-inference"
  type        = string
  default     = "solar-panel-inference"
}

variable "s3_bucket_name" {
  description = "solar-pv-panel-inference-data"
  type        = string
  default     = "solar-panel-inference-data"
}
variable "key_pair_name" {
  description = "Name of the EC2 key pair for batch compute instances"
  type        = string
  default     = null
}
variable "ecr_image_url" {
  description = "URL of the ECR image for batch job"
  type        = string
  default     = "309379105600.dkr.ecr.ap-southeast-2.amazonaws.com/solar-inference:latest"
}
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "ap-southeast-2"
}
variable "model_path" {
  description = "S3 path to the ML model"
  type        = string
  default     = "s3://solar-panel-inference-data/models/EDW_T99_A17.pth"
}
variable "roboflow_model_id" {
  description = "Roboflow model ID (fallback if SSM not available)"
  type        = string
  default     = ""
}

variable "roboflow_api_key" {
  description = "Roboflow API key (fallback if SSM not available)"
  type        = string
  default     = ""
  sensitive   = true
}