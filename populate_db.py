import os
import uuid
from datetime import datetime

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

from config import config
from db import get_connection, init_db

# Initialize DB
init_db()

# Haar cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
FACE_SIZE = (128, 128)

def encode_face_array(face_image: np.ndarray) -> str:
    """Normalize and encode a face crop into a base64 string."""
    import base64
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FACE_SIZE)
    return base64.b64encode(resized.tobytes()).decode("utf-8")

def add_user_to_db(name, passport_number="N/A", nationality="Unknown", date_of_birth="2000-01-01"):
    user_id = str(uuid.uuid4())
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (user_id, name, passport_number, nationality, date_of_birth)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, name, passport_number, nationality, date_of_birth))
    return user_id

def add_face_record(user_id, image_path, face_encoding, confidence_score=1.0, location="LFW Dataset", status="active"):
    record_id = str(uuid.uuid4())
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO face_records (record_id, user_id, image_path, face_encoding, confidence_score, location, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (record_id, user_id, image_path, face_encoding, confidence_score, location, status))

# Load LFW dataset
print("Loading LFW dataset...")
lfw = fetch_lfw_people(min_faces_per_person=70, resize=1.0, color=True)
images = lfw.images  # Shape: (n_samples, height, width, channels)
target_names = lfw.target_names
targets = lfw.target

print(f"Loaded {len(images)} images of {len(target_names)} people.")

# Create directory for images if not exists
os.makedirs("lfw_images", exist_ok=True)

# Group images by person
person_images = {}
for i, target in enumerate(targets):
    name = target_names[target]
    if name not in person_images:
        person_images[name] = []
    person_images[name].append(images[i])

# Add each person to DB
for name, imgs in person_images.items():
    print(f"Adding {name} with {len(imgs)} images...")
    user_id = add_user_to_db(name)
    
    for idx, img in enumerate(imgs):
        # Convert to uint8 if needed
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        # Detect face (simple: assume whole image is face since LFW is cropped)
        # For simplicity, use the whole image as face
        face_encoding = encode_face_array(img_uint8)
        
        # Save image
        image_path = f"lfw_images/{name}_{idx}.jpg"
        cv2.imwrite(image_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        
        # Add to DB
        add_face_record(user_id, image_path, face_encoding)

# Now add user's image
print("Adding your image...")
user_name = "You"  # Or whatever
user_id = add_user_to_db(user_name)

# Assume captured_face.jpg is your image
your_image_path = "captured_face.jpg"
if os.path.exists(your_image_path):
    img = cv2.imread(your_image_path)
    if img is not None:
        face_encoding = encode_face_array(img)
        add_face_record(user_id, your_image_path, face_encoding, location="User Upload")
        print("Your image added.")
    else:
        print("Could not load your image.")
else:
    print("Your image not found.")

print("DB populated successfully!")