from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.parse
import urllib.request

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


def _blob_token() -> str:
    return os.environ.get("BLOB_READ_WRITE_TOKEN", "").strip()


def _is_blob(key: str | None) -> bool:
    return bool(key) and key.startswith("https://") and ".blob.vercel-storage.com" in key


def ensure_bucket() -> None:
    if _blob_token():
        return
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
    if _is_blob(key):
        request = urllib.request.Request(key, headers={"Authorization": f"Bearer {_blob_token()}"})
        with urllib.request.urlopen(request) as response, open(dest, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        return
    s3.download_file(S3_BUCKET, key, dest)


def upload(path: str, key: str, content_type: str = "video/mp4") -> str:
    token = _blob_token()
    if token:
        query = urllib.parse.urlencode({"pathname": key})
        size = os.path.getsize(path)
        connection = http.client.HTTPSConnection("vercel.com", timeout=60 * 60)
        connection.putrequest("PUT", f"/api/blob/?{query}")
        connection.putheader("Authorization", f"Bearer {token}")
        connection.putheader("x-content-type", content_type)
        connection.putheader("x-vercel-blob-access", "private")
        connection.putheader("x-allow-overwrite", "1")
        connection.putheader("Content-Length", str(size))
        connection.endheaders()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                connection.send(chunk)
        response = connection.getresponse()
        raw = response.read()
        if response.status >= 400:
            raise RuntimeError(f"Could not store the cut ({response.status}). {raw[:300].decode('utf-8', 'replace')}")
        payload = json.loads(raw.decode("utf-8"))
        stored = payload.get("url")
        if not isinstance(stored, str) or not stored:
            raise RuntimeError("Storage did not return a file address.")
        return stored
    s3.upload_file(path, S3_BUCKET, key, ExtraArgs={"ContentType": content_type})
    return key


def delete(key: str | None) -> None:
    if not key:
        return
    if _is_blob(key):
        token = _blob_token()
        if not token:
            return
        request = urllib.request.Request(
            "https://vercel.com/api/blob/delete",
            data=json.dumps({"urls": [key]}).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {token}", "content-type": "application/json"},
        )
        try:
            urllib.request.urlopen(request).close()
        except urllib.error.HTTPError:
            return
        return
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
    except ClientError:
        return


def abort_upload(key: str | None, upload_id: str | None) -> None:
    if not key or not upload_id or upload_id == "blob" or _is_blob(key):
        return
    try:
        s3.abort_multipart_upload(Bucket=S3_BUCKET, Key=key, UploadId=upload_id)
    except ClientError:
        return


def exists(key: str | None) -> bool:
    if not key:
        return False
    if _is_blob(key):
        request = urllib.request.Request(key, method="HEAD", headers={"Authorization": f"Bearer {_blob_token()}"})
        try:
            urllib.request.urlopen(request).close()
            return True
        except urllib.error.HTTPError:
            return False
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False
