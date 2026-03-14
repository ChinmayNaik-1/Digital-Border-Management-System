import os
import socket
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _resolve_host(host: str, fallback: str = "localhost") -> str:
    """Return a reachable host name, defaulting to localhost when DNS fails."""
    try:
        socket.getaddrinfo(host, None)
        return host
    except Exception:
        return fallback


def _resolve_endpoint(endpoint: str, fallback_host: str = "localhost") -> str:
    """Resolve endpoint host fallback while preserving port."""
    if not endpoint:
        return endpoint
    if ":" not in endpoint:
        return _resolve_host(endpoint, fallback_host)

    host, sep, port = endpoint.partition(":")
    return f"{_resolve_host(host, fallback_host)}{sep}{port}"


@dataclass(frozen=True)
class Config:
    # Flask
    FLASK_RUN_HOST: str = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    FLASK_RUN_PORT: int = int(os.getenv("FLASK_RUN_PORT", "5000"))
    FLASK_ENV: str = os.getenv("FLASK_ENV", "production")
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "0") in ("1", "true", "True")

    # Postgres
    # Allow either POSTGRES_* or DB_* env vars for compatibility with different examples.
    POSTGRES_HOST: str = _resolve_host(os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST") or "db")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT") or "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB") or os.getenv("DB_NAME") or "border_db"
    POSTGRES_USER: str = os.getenv("POSTGRES_USER") or os.getenv("DB_USER") or "border_user"
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASSWORD") or "border_pass"

    # MinIO
    MINIO_ENDPOINT: str = _resolve_endpoint(os.getenv("MINIO_ENDPOINT", "minio:9000"))
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "faces")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false") in ("1", "true", "True")

    # App
    APP_LOG_LEVEL: str = os.getenv("APP_LOG_LEVEL", "INFO")


config = Config()
