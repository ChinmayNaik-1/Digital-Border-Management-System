import cv2
import numpy as np
import os
import sqlite3
from sklearn.datasets import fetch_olivetti_faces
from scipy.spatial.distance import cosine

# Database setup
DB_PATH = os.path.join(os.path.dirname(__file__), 'faces.db')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'db_images')
os.makedirs(IMAGES_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            image_path TEXT,
            capture_date TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY,
            face_id INTEGER,
            similarity REAL,
            match_date TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (face_id) REFERENCES faces (id)
        )
    ''')
    conn.commit()
    return conn

def save_to_db(conn, name, embedding, image=None):
    cursor = conn.cursor()
    embedding_bytes = embedding.tobytes()
    image_path = None
    if image is not None:
        image_filename = f"{name}_{int(conn.execute('SELECT COUNT(*) FROM faces').fetchone()[0])}.jpg"
        image_path = os.path.join(IMAGES_DIR, image_filename)
        cv2.imwrite(image_path, image)
    cursor.execute('INSERT INTO faces (name, embedding, image_path) VALUES (?, ?, ?)',
                   (name, embedding_bytes, image_path))
    conn.commit()
    return cursor.lastrowid

def load_from_db(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, embedding, image_path FROM faces')
    faces = []
    for row in cursor.fetchall():
        face_id, name, embedding_bytes, image_path = row
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        faces.append((face_id, name, embedding, image_path))
    return faces

def save_match(conn, face_id, similarity):
    cursor = conn.cursor()
    cursor.execute('INSERT INTO matches (face_id, similarity) VALUES (?, ?)', (face_id, similarity))
    conn.commit()

# Function to capture frame from webcam (from app.py)
def _capture_frame(timeout_seconds: int = 5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Ensure /dev/video0 is mounted into the container.")

    end_time = cv2.getTickCount() / cv2.getTickFrequency() + timeout_seconds
    frame = None

    while cv2.getTickCount() / cv2.getTickFrequency() < end_time:
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

# Live webcam capture with preview
def capture_with_preview():
    print("Opening webcam for live preview...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return None

    print("Live preview: Press 'C' to capture, 'Q' to quit.")
    captured = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Live Webcam - Press C to Capture, Q to Quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord('C'):
            captured = frame.copy()
            print("✅ Image captured!")
            break
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured

# Better preprocessing
def preprocess_image(img):
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # For equalizeHist, need uint8
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)
    # Equalize histogram for better contrast
    eq = cv2.equalizeHist(gray)
    return eq

# Load Olivetti faces dataset (smaller, faster to load)
print("Loading Olivetti faces dataset (400 images, 40 people)...")
faces = fetch_olivetti_faces()
images = faces.images  # Shape: (400, 64, 64), grayscale
targets = faces.target  # 0-39

FACE_SIZE = (64, 64)  # Match the dataset size

def encode_face_array(face_image: np.ndarray) -> np.ndarray:
    """Encode a face crop into a normalized array."""
    processed = preprocess_image(face_image)
    resized = cv2.resize(processed, FACE_SIZE)
    # Normalize to 0-1
    normalized = resized.astype(np.float32) / 255.0
    return normalized.flatten()  # Flatten for cosine distance

# Encode all Olivetti faces
olivetti_encodings = []
olivetti_names = []
for i, img in enumerate(images):
    encoding = encode_face_array(img)
    olivetti_encodings.append(encoding)
    olivetti_names.append(f"Olivetti_{targets[i]}")

olivetti_encodings = np.array(olivetti_encodings)

# Init DB and load stored faces
conn = init_db()
db_faces = load_from_db(conn)
db_encodings = []
db_names = []
db_images = []
for face_id, name, embedding, image_path in db_faces:
    db_encodings.append(embedding)
    db_names.append(name)
    db_images.append(image_path)

if db_encodings:
    db_encodings = np.array(db_encodings)
    # Combine Olivetti and DB
    all_encodings = np.vstack([olivetti_encodings, db_encodings])
    all_names = olivetti_names + db_names
else:
    all_encodings = olivetti_encodings
    all_names = olivetti_names

# Capture with live preview
print("Starting live webcam capture...")
captured_img = capture_with_preview()
if captured_img is None:
    print("No image captured.")
    conn.close()
    exit(1)

# Encode the captured image
your_test_encoding = encode_face_array(captured_img)
print("✅ Image encoded.")

# Optionally save to DB
save_choice = input("Save this face to database? (y/n): ").strip().lower()
if save_choice == 'y':
    name = input("Enter name for this face: ").strip()
    if name:
        save_to_db(conn, name, your_test_encoding, captured_img)
        print(f"✅ Face saved as {name}.")
    else:
        print("❌ No name provided, not saved.")

# Compute cosine distances (lower is better)
distances = np.array([cosine(your_test_encoding, enc) for enc in all_encodings])

# Find the best match
min_idx = np.argmin(distances)
min_distance = distances[min_idx]
matched_person = all_names[min_idx]

print(f"Best match: {matched_person} with cosine distance {min_distance:.4f}")
print(f"Matched index: {min_idx}")

# Threshold for match (lower is better match, adjusted to 0.5 for cosine)
threshold = 0.5  # Cosine distance < 0.5 means similar
if min_distance < threshold:
    print("✅ Face recognized!")
    # Save match to DB if it's a DB face
    if min_idx >= len(olivetti_names):
        db_idx = min_idx - len(olivetti_names)
        face_id = db_faces[db_idx][0]
        save_match(conn, face_id, 1 - min_distance)  # Save similarity
else:
    print("❌ Face not recognized.")

# Visualize the match
print("Showing captured image and matched face...")
cv2.imshow("Captured Image", captured_img)

if min_idx < len(olivetti_names):
    # Matched from Olivetti
    matched_img = images[min_idx]
    matched_display = (matched_img * 255).astype(np.uint8)
    cv2.imshow("Matched Face (Olivetti)", matched_display)
else:
    # Matched from DB
    db_idx = min_idx - len(olivetti_names)
    image_path = db_images[db_idx]
    if image_path and os.path.exists(image_path):
        matched_img = cv2.imread(image_path)
        cv2.imshow("Matched Face (DB)", matched_img)
    else:
        print("No image available for matched DB face.")

cv2.waitKey(0)
cv2.destroyAllWindows()

conn.close()