import cv2

def capture_face():
    print("Opening webcam to capture face...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Cannot open webcam.")
        return

    print("Press SPACE to capture, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Spacebar
            cv2.imwrite("captured_face.jpg", frame)
            print("✅ Face captured and saved as captured_face.jpg")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_face()