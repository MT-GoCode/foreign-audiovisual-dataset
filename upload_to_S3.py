
import boto3
import pandas as pd
from io import StringIO
import os
import argparse

# INITIALIZATION

parser = argparse.ArgumentParser(description='Process AWS credentials and bucket name.')
parser.add_argument('--aws_access_key_id', type=str, required=True, help='AWS access key ID')
parser.add_argument('--aws_secret_access_key', type=str, required=True, help='AWS secret access key')
parser.add_argument('--region_name', type=str, required=True, help='AWS region name')
parser.add_argument('--bucket_name', type=str, required=True, help='S3 bucket name')

args = parser.parse_args()


session = boto3.Session(
    aws_access_key_id=args.aws_access_key_id,
    aws_secret_access_key=args.aws_secret_access_key,
    region_name=args.region_name
)

# Create an S3 client
s3 = session.client('s3')

bucket_name = args.bucket_name

folder_map_local_to_s3 = {
    "DATA/video_clips/" : "S3_DATASET/DATA/video_clips",
    "DATA/raw_videos/" : "S3_DATASET/DATA/raw_videos",
    "STATISTICS/" : "S3_DATASET/STATISTICS",
}

csv_map_local_to_s3 = {
    "CSV/raw_videos.csv": "S3_DATASET/CSV/raw_videos.csv",
    "CSV/video_queries.csv": "S3_DATASET/CSV/video_queries.csv",
    "CSV/video_clips.csv": "S3_DATASET/CSV/video_clips.csv",
}

# FUNCTIONS

def verify_s3_structure(s3, bucket_name):
    # Define the expected structure
    expected_files = {
        'S3_DATASET/CSV/raw_videos.csv',
        'S3_DATASET/CSV/video_clips.csv',
        'S3_DATASET/CSV/video_queries.csv',
        'S3_DATASET/DATA/raw_videos/filler',
        'S3_DATASET/DATA/video_clips/filler',
        'S3_DATASET/STATISTICS/filler',
    }

    # Get the list of all objects in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name)
    files_in_bucket = set()
    directories_in_bucket = set()

    # Process the returned objects
    for obj in response.get('Contents', []):
        object_key = obj['Key']
        if object_key.endswith('/'):
            directories_in_bucket.add(object_key)
        else:
            files_in_bucket.add(object_key)

    # print(directories_in_bucket)

    # Check if all expected files and directories are present
    missing_files = expected_files - files_in_bucket
    # missing_directories = expected_directories - directories_in_bucket

    if not missing_files: # and not missing_directories:
        print("Bucket structure is as expected.")
    else:
        if missing_files:
            print("S3 Missing files:", missing_files)

def list_files_in_s3_directory(bucket_name, prefix, s3):
    paginator = s3.get_paginator('list_objects_v2')
    files_in_s3 = set()
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            file_name = obj['Key'].split('/')[-1]
            if file_name:  # Exclude empty names, which occur if the key ends with '/'
                files_in_s3.add(file_name)

    return files_in_s3

def list_files_in_local_directory(local_dir):
    return {file for file in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, file))}

def compare_file_sets(set1, set2):
    unique_to_set1 = set1 - set2
    unique_to_set2 = set2 - set1
    return unique_to_set1, unique_to_set2

def upload_files_to_s3(filenames, local_dir, bucket_name, s3_prefix, s3):
    for file_name in filenames:
        local_path = os.path.join(local_dir, file_name)
        s3_path = f"{s3_prefix}/{file_name}" if s3_prefix else file_name
        s3.upload_file(local_path, bucket_name, s3_path)
        print(f"Uploaded {file_name} to {s3_path} in bucket {bucket_name}")

# EXECUTION

# step 1: check file structure
verify_s3_structure(s3, bucket_name)

for local_folder, s3_folder in folder_map_local_to_s3.items():
    # print(key, value)

    # step 2: 
    FILES_ON_S3 = list_files_in_s3_directory(bucket_name = bucket_name, prefix = s3_folder, s3 = s3)

    FILES_ON_LOCAL = list_files_in_local_directory(local_folder)

    unique_to_s3, unique_to_local = compare_file_sets(FILES_ON_S3, FILES_ON_LOCAL)

    upload_files_to_s3(filenames = unique_to_local,
                       local_dir=local_folder,
                       bucket_name=bucket_name,
                       s3_prefix=s3_folder,
                       s3=s3)

# CSV UPDATE

def read_csv_from_s3(s3, bucket_name, file_key):
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    
    csv_string = response['Body'].read().decode('utf-8')
    df_s3 = pd.read_csv(StringIO(csv_string))
    return df_s3

def write_df_to_s3(s3, df, bucket_name, file_key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Body=csv_buffer.getvalue(), Key=file_key)

def update_s3_csv_with_local_data(s3, local_csv_path, s3_file, bucket_name):
    
    df_local = pd.read_csv(local_csv_path)
    
    df_s3 = read_csv_from_s3(s3, bucket_name, s3_file)
    
    if not df_s3.empty:
        df_combined = pd.concat([df_s3, df_local]).drop_duplicates(keep=False)
    else:
        df_combined = df_local

    write_df_to_s3(s3, df_combined, bucket_name, s3_file)

    print("updated csv: " + local_csv_path + s3_file)

# Usage
print('CSV UPDATE TIME')


for local_csv, s3_csv in csv_map_local_to_s3.items():
    update_s3_csv_with_local_data(s3, local_csv, s3_csv, bucket_name, )