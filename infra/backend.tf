terraform {
  backend "s3" {
    bucket         = "teleco-churn-tfstate-bucket"
    key            = "teleco-customer-churn/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "teleco-churn-tfstate-lock"
    encrypt        = true
  }
}
