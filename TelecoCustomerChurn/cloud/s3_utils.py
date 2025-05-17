import os
import boto3
from botocore.exceptions import NoCredentialsError
import glob

def upload_folder_to_s3(local_folder, bucket, s3_prefix):
    s3 = boto3.client('s3')
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = os.path.join(s3_prefix, relative_path).replace('\\', '/')
            try:
                s3.upload_file(local_path, bucket, s3_path)
                print(f"Uploaded {local_path} to s3://{bucket}/{s3_path}")
            except NoCredentialsError:
                print("AWS credentials not found.")
                raise

def download_folder_from_s3(bucket, s3_prefix, local_folder):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            s3_path = obj['Key']
            rel_path = os.path.relpath(s3_path, s3_prefix)
            local_path = os.path.join(local_folder, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, s3_path, local_path)
            print(f"Downloaded s3://{bucket}/{s3_path} to {local_path}")
