import logging

from minio import Minio
from minio.error import S3Error
from urllib3 import PoolManager, Timeout

from config import config

logger = logging.getLogger(__name__)


def _build_http_client() -> PoolManager:
    # Keep the MinIO connection attempts bounded so the API can start even when
    # the MinIO service is not available.
    timeout = Timeout(connect=2.0, read=4.0)
    return PoolManager(timeout=timeout)


def get_minio_client() -> Minio:
    secure = config.MINIO_SECURE
    endpoint = config.MINIO_ENDPOINT
    return Minio(
        endpoint,
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        secure=secure,
        http_client=_build_http_client(),
    )


def ensure_bucket(client: Minio, bucket_name: str):
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info("Created MinIO bucket '%s'", bucket_name)
        else:
            logger.debug("MinIO bucket '%s' already exists", bucket_name)
    except S3Error as e:
        logger.exception("Failed to ensure bucket %s: %s", bucket_name, e)
        raise
