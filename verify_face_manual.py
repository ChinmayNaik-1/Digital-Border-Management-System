import os
import sys
from datetime import datetime

import cv2
import numpy as np


class FaceVerifier:
    def __init__(self, reference_image_path: str):
        self.reference_image_path = reference_image_path
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.captured_frame = None
        self.reference_image = None
        self.captured_face_region = None
        self.reference_face_region = None

    def validate_reference_image(self) -> bool:
        """Check if reference image exists and can be loaded."""
        if not os.path.exists(self.reference_image_path):
            print(f"❌ Error: Reference image not found at '{self.reference_image_path}'")
            return False

        self.reference_image = cv2.imread(self.reference_image_path)
        if self.reference_image is None:
            print("❌ Error: Could not load reference image. Check if it's a valid image file.")
            return False

        print(f"✅ Reference image loaded: {self.reference_image_path}")
        print(f"   Dimensions: {self.reference_image.shape}")
        return True

    def open_webcam(self) -> bool:
        """Open webcam and capture a frame when SPACE is pressed."""
        print("\n📷 Opening webcam...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Cannot open webcam. Make sure camera is connected and not in use.")
            return False

        print("✅ Webcam opened successfully")
        print("📍 Instructions:")
        print("   - Press SPACEBAR to capture face")
        print("   - Press Q to quit without capturing")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("❌ Failed to read from webcam")
                cap.release()
                return False

            # Mirror for a more natural webcam view
            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                "Press SPACE to capture, Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Webcam - Press SPACE to capture, Q to quit", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                self.captured_frame = frame
                print("✅ Face captured!")
                cv2.destroyAllWindows()
                cap.release()
                return True
            if key == ord("q") or key == ord("Q"):
                print("❌ Capture cancelled")
                cv2.destroyAllWindows()
                cap.release()
                return False

    def detect_face(self, image: np.ndarray, image_name: str = "Image") -> np.ndarray | None:
        """Detect the largest face in the image using Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            print(f"❌ {image_name}: No face detected")
            return None

        if len(faces) > 1:
            print(f"⚠️  {image_name}: Multiple faces detected ({len(faces)}), using largest")
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

        x, y, w, h = faces[0]
        face_region = image[y : y + h, x : x + w]
        print(f"✅ {image_name}: Face detected at ({x}, {y}) with size {w}x{h}")
        return face_region

    def extract_faces(self) -> bool:
        """Extract face regions from both reference and captured images."""
        print("\n🔍 Detecting faces in images...")

        self.reference_face_region = self.detect_face(self.reference_image, "Reference image")
        if self.reference_face_region is None:
            return False

        self.captured_face_region = self.detect_face(self.captured_frame, "Captured frame")
        if self.captured_face_region is None:
            return False

        cv2.imwrite("reference_face_region.jpg", self.reference_face_region)
        cv2.imwrite("captured_face_region.jpg", self.captured_face_region)
        print("✅ Face regions extracted and saved")
        return True

    def compare_faces(self) -> tuple[float, float]:
        """Compare faces using Euclidean distance and return (confidence, distance)."""
        print("\n📊 Comparing faces...")

        size = (128, 128)
        ref_resized = cv2.resize(self.reference_face_region, size)
        cap_resized = cv2.resize(self.captured_face_region, size)

        ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
        cap_gray = cv2.cvtColor(cap_resized, cv2.COLOR_BGR2GRAY)

        diff = ref_gray.astype(np.float32) - cap_gray.astype(np.float32)
        distance = float(np.linalg.norm(diff))

        # Normalize to 0-255 range per pixel (root mean square difference)
        rms = distance / np.sqrt(ref_gray.size)
        normalized = min(max(rms, 0.0), 255.0)

        confidence = max(0.0, 100.0 - (normalized / 255.0 * 100.0))

        return confidence, distance

    def display_results(self, confidence: float, distance: float):
        print("\n" + "=" * 60)
        print("BORDER PROTECTION - FACE VERIFICATION")
        print("=" * 60)
        print(f"\nReference Image: {self.reference_image_path}")
        print("Captured Frame: captured_face.jpg")

        print("\nFace Detection Results:")
        print("- Reference image: ✓ Face detected")
        print("- Webcam frame: ✓ Face detected")

        print("\nComparison Results:")
        print(f"- Confidence Score: {confidence:.2f}%")
        print(f"- Euclidean Distance: {distance:.2f}")
        result = "✅ MATCHED" if confidence > 70 else "❌ NOT MATCHED"
        print(f"- Result: {result}")

        print("\nFiles saved:")
        print("- captured_face.jpg (full webcam frame)")
        print("- captured_face_region.jpg (extracted face from webcam)")
        print("- reference_face_region.jpg (extracted face from reference)")
        print("=" * 60)

    def save_results_to_file(self, confidence: float, distance: float):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("verification_result.txt", "w", encoding="utf-8") as f:
            f.write("BORDER PROTECTION - FACE VERIFICATION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Reference Image: {self.reference_image_path}\n")
            f.write(f"Confidence Score: {confidence:.2f}%\n")
            f.write(f"Euclidean Distance: {distance:.2f}\n")
            f.write(f"Result: {'MATCHED' if confidence > 70 else 'NOT MATCHED'}\n")
            f.write("=" * 60 + "\n")

        print("✅ Results saved to verification_result.txt")

    def show_comparison_images(self):
        print("\n🖼️  Displaying comparison images... (Press any key to close)")

        ref_img = cv2.imread("reference_face_region.jpg")
        cap_img = cv2.imread("captured_face_region.jpg")

        if ref_img is None or cap_img is None:
            print("⚠️  Unable to load comparison images for display.")
            return

        h1, w1 = ref_img.shape[:2]
        h2, w2 = cap_img.shape[:2]
        max_h = max(h1, h2)

        ref_resized = cv2.resize(ref_img, (int(w1 * max_h / h1), max_h))
        cap_resized = cv2.resize(cap_img, (int(w2 * max_h / h2), max_h))

        comparison = np.hstack([ref_resized, cap_resized])

        cv2.putText(comparison, "Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(
            comparison,
            "Captured",
            (ref_resized.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Face Comparison", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def verify(self) -> bool:
        print("\n" + "=" * 60)
        print("BORDER PROTECTION - FACE VERIFICATION SYSTEM")
        print("=" * 60)

        if not self.validate_reference_image():
            return False

        if not self.open_webcam():
            return False

        cv2.imwrite("captured_face.jpg", self.captured_frame)
        print("✅ Captured frame saved to captured_face.jpg")

        if not self.extract_faces():
            return False

        confidence, distance = self.compare_faces()
        self.display_results(confidence, distance)
        self.save_results_to_file(confidence, distance)
        self.show_comparison_images()

        return confidence > 70


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_face_manual.py <reference_image_path>")
        print("Example: python verify_face_manual.py my_face.jpg")
        sys.exit(1)

    reference_image_path = sys.argv[1]
    verifier = FaceVerifier(reference_image_path)
    success = verifier.verify()

    if success:
        print("✅ VERIFICATION PASSED - Face matched!")
        sys.exit(0)
    else:
        print("❌ VERIFICATION FAILED - Face did not match")
        sys.exit(1)


if __name__ == "__main__":
    main()
