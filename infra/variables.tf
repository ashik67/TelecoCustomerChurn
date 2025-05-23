# variables.tf
# Terraform variable definitions for AWS infrastructure

variable "aws_region" {
  description = "AWS region to deploy resources in"
  type        = string
  default     = "us-east-1"  # Default region
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for artifacts"
  type        = string
}

variable "ecr_repo_name" {
  description = "Name of the ECR repository"
  type        = string
}

variable "cloudwatch_log_group" {
  description = "Name of the CloudWatch log group"
  type        = string
  default     = "/teleco-churn/logs"
}

variable "ec2_role_name" {
  description = "Name of the IAM role for EC2"
  type        = string
  default     = "teleco-churn-ec2-role"
}

variable "ec2_instance_profile_name" {
  description = "Name of the IAM instance profile for EC2"
  type        = string
  default     = "teleco-churn-ec2-profile"
}

variable "ec2_sg_name" {
  description = "Name of the EC2 security group"
  type        = string
  default     = "teleco-churn-sg"
}

variable "ami_id" {
  description = "AMI ID for the EC2 instance"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t2.micro"
}

variable "key_name" {
  description = "Name of the EC2 key pair for SSH access"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for the EC2 instance"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for the security group"
  type        = string
}
