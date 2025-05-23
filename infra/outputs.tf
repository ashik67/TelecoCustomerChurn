# outputs.tf
# Terraform outputs for AWS infrastructure resources

output "s3_bucket_name" {
  value = aws_s3_bucket.artifacts.bucket
  description = "Name of the S3 bucket for artifacts"
}

output "ecr_repo_url" {
  value = aws_ecr_repository.app_repo.repository_url
  description = "URL of the ECR repository for Docker images"
}

output "cloudwatch_log_group_name" {
  value = aws_cloudwatch_log_group.app_logs.name
  description = "Name of the CloudWatch log group"
}

output "ec2_instance_id" {
  value = aws_instance.app_server.id
  description = "ID of the EC2 instance running the app"
}

output "ec2_public_dns" {
  value = aws_instance.app_server.public_dns
  description = "Public DNS of the EC2 instance"
}
