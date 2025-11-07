terraform {
  backend "s3" {
    bucket         = "solar-panel-inference-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "ap-southeast-2"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}