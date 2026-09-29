from __future__ import annotations

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from config import (
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT,
    S3_REGION,
    S3_SECRET_KEY,
    WEB_ORIGINS,
)

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
    config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 3}),
)


def ensure_bucket() -> None:
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=S3_BUCKET)
    s3.put_bucket_cors(
        Bucket=S3_BUCKET,
        CORSConfiguration={
            "CORSRules": [
                {
                    "AllowedOrigins": WEB_ORIGINS,
                    "AllowedMethods": ["GET", "PUT", "HEAD"],
                    "AllowedHeaders": ["*"],
                    "ExposeHeaders": ["ETag", "Content-Length", "Content-Range", "Accept-Ranges"],
                    "MaxAgeSeconds": 3600,
                }
            ]
        },
    )


def download(key: str, dest: str) -> None:
    s3.download_file(S3_BUCKET, key, dest)


def upload(path: str, key: str, content_type: str = "video/mp4") -> None:
    s3.upload_file(path, S3_BUCKET, key, ExtraArgs={"ContentType": content_type})


def delete(key: str | None) -> None:
    if not key:
        return
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
    except ClientError:
        return


def abort_upload(key: str | None, upload_id: str | None) -> None:
    if not key or not upload_id:
        return
    try:
        s3.abort_multipart_upload(Bucket=S3_BUCKET, Key=key, UploadId=upload_id)
    except ClientError:
        return


def exists(key: str | None) -> bool:
    if not key:
        return False
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False
