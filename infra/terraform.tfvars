# terraform.tfvars
# Example values for Terraform variables (edit as needed for your environment)

aws_region = "us-east-1"  # AWS region
s3_bucket_name = "teleco-churn-artifacts"  # S3 bucket name for artifacts
ecr_repo_name = "teleco-churn-fastapi"     # ECR repository name
cloudwatch_log_group = "/teleco-churn/logs"  # CloudWatch log group name
ec2_role_name = "teleco-churn-ec2-role"      # IAM role name for EC2
ec2_instance_profile_name = "teleco-churn-ec2-profile"  # Instance profile name
ec2_sg_name = "teleco-churn-sg"              # Security group name
ami_id = "ami-0c02fb55956c7d316"             # AMI ID for EC2 instance (Amazon Linux 2)
instance_type = "t2.micro"                    # EC2 instance type
key_name = "my-ec2-key-pair"                 # EC2 key pair name (set in AWS & GitHub secret)
# subnet_id and vpc_id removed; now dynamically fetched
