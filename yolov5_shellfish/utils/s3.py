"""
S3 utility functions
"""
from io import BytesIO

import boto3
import botocore


def s3_list_blobs(bucket_name, prefix):
    """List blob paths in prefix folder.

    Args:
        bucket_name (str)
        prefix (str)

    Returns:
        list(str): list of blob paths
    """
    s3 = boto3.resource("s3")
    s3_bucket = s3.Bucket(bucket_name)
    return [f.key for f in s3_bucket.objects.filter(Prefix=prefix).all()]


def s3_download_file(bucket_name, origin_blob_path, dest_filename):
    """Download blob from S3 bucket.

    Args:
        bucket_name (str)
        origin_blob_path (str)
        dest_filename (str): destination filename
    """
    s3 = boto3.resource("s3")

    try:
        s3.Bucket(bucket_name).download_file(origin_blob_path, dest_filename)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise


def s3_upload_file(origin_file_path, bucket_name, dest_key):
    """Upload file to S3 bucket.

    Args:
        origin_file_path (str)
        bucket_name (str)
        dest_key (str)
    """
    s3 = boto3.resource("s3")

    try:
        s3.Object(bucket_name, dest_key).upload_file(origin_file_path)
    except botocore.exceptions.ClientError:
        raise


def s3_get_blob(bucket_name, blob_path):
    """Get blob.

    Args:
        bucket_name (str)
        blob_path (str)

    Returns:
        bytes
    """
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_name, blob_path)
    return obj.get()["Body"].read()


def s3_txt_read(bucket_name, blob_path):
    """Load text file from S3 bucket.

    Args:
        bucket_name (str)
        blob_path (str)

    Returns:
        str
    """
    body = s3_get_blob(bucket_name, blob_path)
    txt = body.decode("utf-8")
    return txt


def s3_imread(bucket_name, blob_path):
    """Load image from S3 as array.

    Args:
        bucket_name (str)
        blob_path (str)

    Returns:
        numpy.array in BGR
    """
    import cv2
    import numpy as np

    body = s3_get_blob(bucket_name, blob_path)
    img_arr = np.asarray(bytearray(body), dtype=np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)  # BGR
    return image


def s3_buffer_read(bucket_name, blob_path):
    """Load file from S3 bucket as buffer.

    Args:
        bucket_name (str)
        blob_path (str)

    Returns:
        BytesIO object
    """
    body = s3_get_blob(bucket_name, blob_path)
    buffered = BytesIO(body)
    return buffered
