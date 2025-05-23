# main.tf
# Terraform AWS provider configuration
provider "aws" {
  region = var.aws_region  # AWS region to deploy resources
}

# S3 bucket for storing ML artifacts
resource "aws_s3_bucket" "artifacts" {
  bucket = var.s3_bucket_name  # Name of the S3 bucket
  force_destroy = true         # Allow bucket deletion with all contents
}

# ECR repository for Docker images
resource "aws_ecr_repository" "app_repo" {
  name = var.ecr_repo_name  # Name of the ECR repository
}

# CloudWatch Log Group for centralized logging
resource "aws_cloudwatch_log_group" "app_logs" {
  name = var.cloudwatch_log_group      # Log group name
  retention_in_days = 14              # Retain logs for 14 days
}

# IAM policy document for EC2 assume role
# Allows EC2 to assume this role
# Used for S3, ECR, and CloudWatch access
# See aws_iam_role below

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

# IAM Role for EC2 instance
resource "aws_iam_role" "ec2_role" {
  name               = var.ec2_role_name
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

# Attach S3, ECR, and CloudWatch policies to EC2 role
resource "aws_iam_role_policy_attachment" "ec2_s3" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}
resource "aws_iam_role_policy_attachment" "ec2_ecr" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess"
}
resource "aws_iam_role_policy_attachment" "ec2_cloudwatch" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# Instance profile for EC2 (required to attach IAM role)
resource "aws_iam_instance_profile" "ec2_profile" {
  name = var.ec2_instance_profile_name
  role = aws_iam_role.ec2_role.name
}

# Fetch the default VPC
data "aws_vpc" "default" {
  default = true
}

# Fetch all subnets in the default VPC
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Security Group for EC2 instance
resource "aws_security_group" "ec2_sg" {
  name        = var.ec2_sg_name
  description = "Allow SSH and HTTP access"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Allow SSH from anywhere (change for production)
  }
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Allow FastAPI app port
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]  # Allow all outbound traffic
  }
}

# EC2 instance for model serving/API
resource "aws_instance" "app_server" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  tags = {
    Name = "teleco-churn-app-server"
  }
}
