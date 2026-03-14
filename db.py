import logging
import psycopg2
import psycopg2.extras

from config import config

logger = logging.getLogger(__name__)


def get_connection():
    # Use a short connection timeout so the app can still start when DB is unavailable.
    return psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        connect_timeout=3,
    )


def init_db():
    """Create required tables if they do not exist."""
    sql = [
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            passport_number TEXT NOT NULL,
            nationality TEXT NOT NULL,
            date_of_birth DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS face_records (
            record_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            image_path TEXT NOT NULL,
            face_encoding TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            location TEXT,
            status TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS verification_logs (
            log_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            face_record_id TEXT,
            verification_status TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            border_gate TEXT
        );
        """,
    ]

    conn = None
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                for statement in sql:
                    cur.execute(statement)
        logger.info("Database initialized")
    except Exception as e:
        logger.exception("Failed to initialize database: %s", e)
        raise
    finally:
        if conn:
            conn.close()
