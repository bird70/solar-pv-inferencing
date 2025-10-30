
provider "aws" {
  region = "ap-southeast-2"  # Auckland, NZ region
}

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
