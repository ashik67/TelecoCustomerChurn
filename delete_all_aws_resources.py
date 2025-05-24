import boto3

def delete_all_resources():
    session = boto3.Session()
    region = session.region_name or 'us-east-1'
    print(f"Deleting AWS resources in region: {region}")

    # S3 Buckets
    s3 = session.client('s3')
    for bucket in s3.list_buckets().get('Buckets', []):
        name = bucket['Name']
        print(f"Deleting S3 bucket: {name}")
        try:
            # Delete all objects
            objects = s3.list_objects_v2(Bucket=name).get('Contents', [])
            for obj in objects:
                s3.delete_object(Bucket=name, Key=obj['Key'])
            s3.delete_bucket(Bucket=name)
        except Exception as e:
            print(f"  Error deleting bucket {name}: {e}")

    # EC2 Instances
    ec2 = session.resource('ec2')
    instances = list(ec2.instances.all())
    if instances:
        print("Terminating EC2 instances...")
        ec2.instances.filter(InstanceIds=[i.id for i in instances]).terminate()

    # ECR Repositories
    ecr = session.client('ecr')
    for repo in ecr.describe_repositories().get('repositories', []):
        repo_name = repo['repositoryName']
        print(f"Deleting ECR repository: {repo_name}")
        try:
            images = ecr.list_images(repositoryName=repo_name).get('imageIds', [])
            if images:
                ecr.batch_delete_image(repositoryName=repo_name, imageIds=images)
            ecr.delete_repository(repositoryName=repo_name, force=True)
        except Exception as e:
            print(f"  Error deleting ECR repo {repo_name}: {e}")

    # Lambda Functions
    lambda_client = session.client('lambda')
    for func in lambda_client.list_functions().get('Functions', []):
        name = func['FunctionName']
        print(f"Deleting Lambda function: {name}")
        try:
            lambda_client.delete_function(FunctionName=name)
        except Exception as e:
            print(f"  Error deleting Lambda {name}: {e}")

    # CloudWatch Log Groups
    logs = session.client('logs')
    for group in logs.describe_log_groups().get('logGroups', []):
        name = group['logGroupName']
        print(f"Deleting CloudWatch log group: {name}")
        try:
            logs.delete_log_group(logGroupName=name)
        except Exception as e:
            print(f"  Error deleting log group {name}: {e}")

    # RDS Instances
    rds = session.client('rds')
    for db in rds.describe_db_instances().get('DBInstances', []):
        dbid = db['DBInstanceIdentifier']
        print(f"Deleting RDS instance: {dbid}")
        try:
            rds.delete_db_instance(DBInstanceIdentifier=dbid, SkipFinalSnapshot=True, DeleteAutomatedBackups=True)
        except Exception as e:
            print(f"  Error deleting RDS {dbid}: {e}")

    print("Resource deletion initiated. Some resources may take time to be fully removed.")

if __name__ == "__main__":
    delete_all_resources()
