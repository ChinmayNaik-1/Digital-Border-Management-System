import base64
import io
import logging
import socket
import threading
import uuid

from datetime import datetime

import cv2
import numpy as np
import psycopg2
import psycopg2.extras
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
from minio.error import S3Error

from config import config
from db import get_connection, init_db
from minio_client import ensure_bucket, get_minio_client


logging.basicConfig(level=getattr(logging, config.APP_LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger(__name__)


# Haar cascade for face detection (bundled with OpenCV)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (128, 128)


def _encode_face_array(face_image: np.ndarray) -> str:
    """Normalize and encode a face crop into a base64 string."""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FACE_SIZE)
    # Normalize to 0-255 range and store as bytes
    return base64.b64encode(resized.tobytes()).decode("utf-8")


def _decode_face_encoding(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr.reshape(FACE_SIZE)


def _compute_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Euclidean distance for grayscale pixel vectors
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def _capture_frame(timeout_seconds: int = 5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Ensure /dev/video0 is mounted into the container.")

    end_time = datetime.utcnow().timestamp() + timeout_seconds
    frame = None

    while datetime.utcnow().timestamp() < end_time:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        # allow the camera to warm up
        if frame.size > 0:
            break

    cap.release()

    if frame is None:
        raise RuntimeError("Failed to capture image from webcam")

    return frame


def _load_image_from_request():
    """Load an image from a multipart/form-data upload or base64 JSON key."""

    # Multipart form upload (e.g., curl -F "image=@./face.jpg")
    if "image" in request.files:
        file = request.files["image"]
        data = file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Uploaded image file is not a valid image")
        return img

    # JSON payload with base64 encoded image
    body = request.get_json(silent=True) or {}
    b64 = body.get("image_base64") or body.get("image")
    if isinstance(b64, str) and b64:
        try:
            decoded = base64.b64decode(b64)
            arr = np.frombuffer(decoded, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError()
            return img
        except Exception:
            raise ValueError("Invalid base64 image data")

    return None


def _detect_face(frame: np.ndarray):
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # choose the largest face
    x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    return frame[y : y + h, x : x + w]


def _save_image_to_minio(client, bucket: str, key: str, image: np.ndarray):
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Failed to encode image as JPEG")
    data = io.BytesIO(buffer.tobytes())
    client.put_object(bucket, key, data, length=data.getbuffer().nbytes, content_type="image/jpeg")
    return f"{bucket}/{key}"


def _build_db_connection():
    return get_connection()


def _dict_from_cursor(cursor):
    return [dict(row) for row in cursor]


def _tcp_service_check(host: str, port: int, timeout: float = 0.8) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Initialize database and object store in the background so the HTTP layer starts fast.
    def _initialize_infrastructure():
        try:
            init_db()
        except Exception:
            logger.exception("Database initialization failed")

        minio_client = get_minio_client()
        try:
            ensure_bucket(minio_client, config.MINIO_BUCKET)
        except Exception:
            logger.exception("MinIO bucket creation failed")

    threading.Thread(target=_initialize_infrastructure, daemon=True).start()

    minio_client = get_minio_client()

    @app.route("/api/health", methods=["GET"])
    def health():
        status = {"status": "ok"}

        # Quick TCP check for DB and MinIO (so health is fast even when services are down)
        db_ok = _tcp_service_check(config.POSTGRES_HOST, config.POSTGRES_PORT)
        status["db"] = "ok" if db_ok else "unavailable"

        minio_host, _, minio_port = config.MINIO_ENDPOINT.partition(":")
        minio_ok = False
        try:
            minio_ok = _tcp_service_check(minio_host, int(minio_port) if minio_port else 9000)
        except ValueError:
            minio_ok = False
        status["minio"] = "ok" if minio_ok else "unavailable"

        return jsonify(status)

    @app.route("/api/users", methods=["POST"])
    def create_user():
        data = request.get_json(force=True)
        required = ["name", "passport_number", "nationality", "date_of_birth"]
        for key in required:
            if key not in data:
                return jsonify({"error": f"Missing field: {key}"}), 400

        user_id = str(uuid.uuid4())
        try:
            with _build_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO users (user_id, name, passport_number, nationality, date_of_birth)
                        VALUES (%s, %s, %s, %s, %s)""",
                        (
                            user_id,
                            data["name"],
                            data["passport_number"],
                            data["nationality"],
                            data["date_of_birth"],
                        ),
                    )
            return jsonify({"user_id": user_id}), 201
        except Exception as e:
            logger.exception("Failed to create user")
            return jsonify({"error": "Failed to create user"}), 500

    @app.route("/api/users/<user_id>", methods=["GET"])
    def get_user(user_id):
        try:
            with _build_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
                    user = cur.fetchone()
            if not user:
                return jsonify({"error": "User not found"}), 404
            return jsonify(user)
        except Exception as e:
            logger.exception("Failed to fetch user")
            return jsonify({"error": "Failed to fetch user"}), 500

    def _capture_and_prepare_face():
        # Allow using an uploaded image (useful when webcam is unavailable)
        frame = _load_image_from_request() or _capture_frame()
        face = _detect_face(frame)
        if face is None:
            raise ValueError("No face detected")
        encoding = _encode_face_array(face)
        return face, encoding

    @app.route("/api/face/capture", methods=["POST"])
    def capture_face():
        payload = request.get_json(silent=True) or {}
        user_id = payload.get("user_id") or request.form.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing required field: user_id"}), 400

        location = payload.get("location") or request.form.get("location")
        status = payload.get("status") or request.form.get("status")

        try:
            face_img, encoding = _capture_and_prepare_face()
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception("Error capturing face")
            return jsonify({"error": "Failed to capture face"}), 500

        record_id = str(uuid.uuid4())
        object_key = f"faces/{record_id}.jpg"

        try:
            image_path = _save_image_to_minio(minio_client, config.MINIO_BUCKET, object_key, face_img)
        except Exception as e:
            logger.exception("Failed to store face image")
            return jsonify({"error": "Failed to store face image"}), 500

        try:
            with _build_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO face_records (record_id, user_id, image_path, face_encoding, confidence_score, location, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (
                            record_id,
                            user_id,
                            image_path,
                            encoding,
                            1.0,
                            location,
                            status,
                        ),
                    )
            return jsonify({"record_id": record_id, "image_path": image_path}), 201
        except Exception as e:
            logger.exception("Failed to persist face record")
            return jsonify({"error": "Failed to persist face record"}), 500

    @app.route("/api/face/verify", methods=["POST"])
    def verify_face():
        payload = request.get_json(silent=True) or {}
        user_id = payload.get("user_id") or request.form.get("user_id")
        if not user_id:
            return jsonify({"error": "Missing required field: user_id"}), 400

        border_gate = payload.get("border_gate") or request.form.get("border_gate")

        try:
            _, probe_encoding = _capture_and_prepare_face()
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception("Error capturing face")
            return jsonify({"error": "Failed to capture face"}), 500

        try:
            with _build_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT record_id, face_encoding FROM face_records WHERE user_id = %s", (user_id,)
                    )
                    records = cur.fetchall()
        except Exception as e:
            logger.exception("Failed to query face records")
            return jsonify({"error": "Failed to query face records"}), 500

        if not records:
            return jsonify({"verified": False, "reason": "No enrollment records"}), 404

        probe_arr = _decode_face_encoding(probe_encoding)
        best = None
        best_distance = None
        for rec in records:
            try:
                enrolled_encoding = _decode_face_encoding(rec["face_encoding"])
                dist = _compute_distance(enrolled_encoding, probe_arr)
            except Exception:
                continue
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best = rec

        if best is None or best_distance is None:
            return jsonify({"verified": False, "reason": "Failed to compute match"}), 500

        # Choose threshold based on empirical experimentation; adjust as needed.
        threshold = 5000.0
        confidence = max(0.0, 1.0 - (best_distance / threshold))
        verified = best_distance < threshold

        log_id = str(uuid.uuid4())
        try:
            with _build_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO verification_logs (log_id, user_id, face_record_id, verification_status, confidence_score, border_gate)
                        VALUES (%s, %s, %s, %s, %s, %s)""",
                        (
                            log_id,
                            user_id,
                            best.get("record_id"),
                            "success" if verified else "failure",
                            float(confidence),
                            border_gate,
                        ),
                    )
        except Exception:
            logger.exception("Failed to create verification log")

        return jsonify(
            {
                "verified": verified,
                "confidence": confidence,
                "matched_record_id": best.get("record_id"),
                "distance": best_distance,
            }
        )

    @app.route("/api/face/records/<user_id>", methods=["GET"])
    def list_face_records(user_id):
        try:
            with _build_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT record_id, image_path, confidence_score, location, status, created_at, updated_at FROM face_records WHERE user_id = %s ORDER BY created_at DESC",
                        (user_id,),
                    )
                    records = cur.fetchall()
            return jsonify(records)
        except Exception as e:
            logger.exception("Failed to list face records")
            return jsonify({"error": "Failed to list face records"}), 500

    @app.route("/api/verification-logs/<user_id>", methods=["GET"])
    def list_verification_logs(user_id):
        try:
            with _build_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT log_id, face_record_id, verification_status, confidence_score, timestamp, border_gate FROM verification_logs WHERE user_id = %s ORDER BY timestamp DESC",
                        (user_id,),
                    )
                    logs = cur.fetchall()
            return jsonify(logs)
        except Exception as e:
            logger.exception("Failed to list verification logs")
            return jsonify({"error": "Failed to list verification logs"}), 500

    @app.route("/api/image/<path:image_path>", methods=["GET"])
    def get_image(image_path):
        try:
            bucket, *key_parts = image_path.split("/", 1)
            if not key_parts:
                return jsonify({"error": "Invalid image path"}), 400
            key = key_parts[0]
            response = minio_client.get_object(bucket, key)
            data = response.read()
            response.close()
            response.release_conn()
            return send_file(io.BytesIO(data), mimetype="image/jpeg")
        except S3Error as e:
            logger.exception("Failed to fetch image from MinIO")
            return jsonify({"error": "Image not found"}), 404
        except Exception as e:
            logger.exception("Failed to fetch image")
            return jsonify({"error": "Failed to fetch image"}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=config.FLASK_RUN_HOST, port=config.FLASK_RUN_PORT, debug=config.FLASK_DEBUG)
