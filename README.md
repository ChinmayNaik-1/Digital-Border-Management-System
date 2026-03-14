# Digital Border Management System (MVP)

A minimal face recognition border protection backend using Flask, PostgreSQL, and MinIO.

## 🚀 Quickstart (Docker)

1) Copy the example env file:

```bash
cp .env.example .env
```

2) Start all services:

```bash
docker compose up --build
```

3) The API will be available at: `http://localhost:5000`

> ✅ **Note:** The app expects a webcam device at `/dev/video0` mounted into the container. On Linux, `docker compose` will mount it automatically via the provided `devices:` entry. On Windows, you can still use the API by uploading an image (see face capture section) instead of using a webcam.
>
> ⚠️ **Local Python runtime:** If you run `python app.py` outside of Docker, use **Python 3.12** (the dependencies are pinned to versions that provide wheels for 3.12). Otherwise pip may try to build packages like numpy from source.

---

## 🧱 API Endpoints

### User Management
- `POST /api/users` – Register a user
- `GET /api/users/<user_id>` – Fetch user info

### Face Enrollment / Verification
- `POST /api/face/capture` – Capture face (webcam or image upload) and store image & metadata
- `POST /api/face/verify` – Capture face (webcam or image upload) and verify against enrolled records
- `GET /api/face/records/<user_id>` – List enrolled face records
- `GET /api/verification-logs/<user_id>` – Audit history of verifications

#### Face capture with uploaded image (no webcam required)

```bash
curl -X POST http://localhost:5000/api/face/capture \
  -F "user_id=<USER_ID>" \
  -F "image=@./face.jpg"
```

You can also send base64-encoded images as JSON:

```bash
curl -X POST http://localhost:5000/api/face/capture \
  -H "Content-Type: application/json" \
  -d '{"user_id":"<USER_ID>", "image_base64":"<BASE64>"}'
```

### Image & Health
- `GET /api/image/<bucket>/<key>` – Retrieve stored image from MinIO
- `GET /api/health` – Health check for DB + MinIO

---

## 📦 Architecture

- **Flask**: REST API
- **PostgreSQL**: Stores users, face records, and verification logs
- **MinIO**: Object store for face images
- **OpenCV**: Webcam capture + Haar Cascade face detection

---

## 🧩 Database Schema

- `users`
- `face_records`
- `verification_logs`

(Table schema defined and auto-created at app startup.)

---

## ⚙️ Configuration

All configuration values are controlled via environment variables (see `.env.example`).

---

## 🧪 Notes / Limitations

- Face matching uses a simplistic pixel-wise encoding (grayscale 128×128 crop). Improving accuracy requires a proper face embedding model (e.g., `face_recognition`, `dlib`, `onnx` models).
- Confidence threshold is hard-coded and may need tuning.

---

## 🧰 Dependencies

See `requirements.txt` for the pinned Python dependency versions.
