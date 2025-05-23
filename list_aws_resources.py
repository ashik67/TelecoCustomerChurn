import boto3

def list_resources():
    session = boto3.Session()
    region = session.region_name or 'us-east-1'
    print(f"Listing AWS resources in region: {region}")

    # S3 Buckets
    s3 = session.client('s3')
    print("\nS3 Buckets:")
    for bucket in s3.list_buckets().get('Buckets', []):
        print(f"  - {bucket['Name']}")

    # EC2 Instances
    ec2 = session.client('ec2')
    print("\nEC2 Instances:")
    for res in ec2.describe_instances().get('Reservations', []):
        for inst in res.get('Instances', []):
            print(f"  - {inst['InstanceId']} ({inst['State']['Name']})")

    # ECR Repositories
    ecr = session.client('ecr')
    print("\nECR Repositories:")
    for repo in ecr.describe_repositories().get('repositories', []):
        print(f"  - {repo['repositoryName']}")

    # IAM Roles
    iam = session.client('iam')
    print("\nIAM Roles:")
    for role in iam.list_roles().get('Roles', []):
        print(f"  - {role['RoleName']}")

    # CloudWatch Log Groups
    logs = session.client('logs')
    print("\nCloudWatch Log Groups:")
    for group in logs.describe_log_groups().get('logGroups', []):
        print(f"  - {group['logGroupName']}")

    # Lambda Functions
    lambda_client = session.client('lambda')
    print("\nLambda Functions:")
    for func in lambda_client.list_functions().get('Functions', []):
        print(f"  - {func['FunctionName']}")

    # RDS Instances
    rds = session.client('rds')
    print("\nRDS Instances:")
    for db in rds.describe_db_instances().get('DBInstances', []):
        print(f"  - {db['DBInstanceIdentifier']}")

if __name__ == "__main__":
    list_resources()
