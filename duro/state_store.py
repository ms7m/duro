"""State store protocol and implementations for large executor state blobs.

When serialized executor state exceeds the inline threshold, it is offloaded
to external storage. Only a reference key is stored in the Temporal payload.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from loguru import logger


class PayloadTooLargeError(Exception):
    """Raised when serialized state exceeds the inline store limit."""

    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(
            f"[state_store] Payload size {size_bytes:,} bytes exceeds "
            f"max inline size {max_bytes:,} bytes. Configure a blob StateStore "
            f"(FileStateStore, S3StateStore, GCSStateStore, R2StateStore) to handle large payloads."
        )


@runtime_checkable
class StateStore(Protocol):
    """Backend for persisting large executor state blobs."""

    async def put(self, key: str, data: bytes) -> None: ...
    async def get(self, key: str) -> bytes: ...
    async def delete(self, key: str) -> None: ...


# * InlineStore (default) — no external storage *


class InlineStore:
    """No external storage. Raises if payload exceeds threshold."""

    def __init__(self, max_inline_bytes: int = 1_500_000) -> None:
        self.max_inline_bytes = max_inline_bytes

    async def put(self, key: str, data: bytes) -> None:
        raise PayloadTooLargeError(len(data), self.max_inline_bytes)

    async def get(self, key: str) -> bytes:
        raise RuntimeError(
            "[state_store] InlineStore does not support get — "
            "this should never be called"
        )

    async def delete(self, key: str) -> None:
        pass  # nothing to delete


# * FileStateStore — local filesystem *


class FileStateStore:
    """Store blobs on the local filesystem. Useful for dev/testing."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe_key = key.replace("/", os.sep)
        return self.directory / safe_key

    async def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.debug(f"[state_store] Wrote {len(data):,} bytes to {path}")

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        data = path.read_bytes()
        logger.debug(f"[state_store] Read {len(data):,} bytes from {path}")
        return data

    async def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
            logger.debug(f"[state_store] Deleted {path}")


# * S3StateStore — Amazon S3 *


class S3StateStore:
    """Store blobs in Amazon S3. Requires `boto3`."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        **boto_kwargs: str,
    ) -> None:
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "boto3 is required for S3StateStore. Install with: uv add boto3"
            ) from e

        self.bucket = bucket
        self.prefix = prefix
        kwargs: dict[str, str] = {}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        kwargs.update(boto_kwargs)
        self._client = boto3.client("s3", **kwargs)

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}" if self.prefix else key

    async def put(self, key: str, data: bytes) -> None:
        s3_key = self._key(key)
        self._client.put_object(Bucket=self.bucket, Key=s3_key, Body=data)
        logger.debug(
            f"[state_store] Wrote {len(data):,} bytes to s3://{self.bucket}/{s3_key}"
        )

    async def get(self, key: str) -> bytes:
        s3_key = self._key(key)
        response = self._client.get_object(Bucket=self.bucket, Key=s3_key)
        data: bytes = response["Body"].read()
        logger.debug(
            f"[state_store] Read {len(data):,} bytes from s3://{self.bucket}/{s3_key}"
        )
        return data

    async def delete(self, key: str) -> None:
        s3_key = self._key(key)
        self._client.delete_object(Bucket=self.bucket, Key=s3_key)
        logger.debug(f"[state_store] Deleted s3://{self.bucket}/{s3_key}")


# * R2StateStore — Cloudflare R2 (S3-compatible) *


class R2StateStore(S3StateStore):
    """Store blobs in Cloudflare R2. Thin wrapper around S3StateStore."""

    def __init__(
        self,
        account_id: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        prefix: str = "",
    ) -> None:
        super().__init__(
            bucket=bucket,
            prefix=prefix,
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )


# * GCSStateStore — Google Cloud Storage *


class GCSStateStore:
    """Store blobs in Google Cloud Storage. Requires `google-cloud-storage`."""

    def __init__(self, bucket: str, prefix: str = "") -> None:
        try:
            from google.cloud import storage  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "google-cloud-storage is required for GCSStateStore. "
                "Install with: uv add google-cloud-storage"
            ) from e

        self.prefix = prefix
        client = storage.Client()
        self._bucket = client.bucket(bucket)

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}" if self.prefix else key

    async def put(self, key: str, data: bytes) -> None:
        blob = self._bucket.blob(self._key(key))
        blob.upload_from_string(data)
        logger.debug(
            f"[state_store] Wrote {len(data):,} bytes to gs://{self._bucket.name}/{self._key(key)}"
        )

    async def get(self, key: str) -> bytes:
        blob = self._bucket.blob(self._key(key))
        data: bytes = blob.download_as_bytes()
        logger.debug(
            f"[state_store] Read {len(data):,} bytes from gs://{self._bucket.name}/{self._key(key)}"
        )
        return data

    async def delete(self, key: str) -> None:
        blob = self._bucket.blob(self._key(key))
        blob.delete()
        logger.debug(
            f"[state_store] Deleted gs://{self._bucket.name}/{self._key(key)}"
        )
