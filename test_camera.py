"""
Simple Camera & Face Detection Test
Run this first to verify your camera works before full setup.

Usage:
    pip install opencv-python
    python test_camera.py
"""

import cv2
import sys

def test_camera():
    print("=" * 50)
    print("  Camera & Face Detection Test")
    print("=" * 50)
    
    # Try to find a working camera
    camera_index = 0
    cap = None
    
    for i in range(4):  # Try cameras 0-3
        print(f"\nTrying camera index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ Camera {i} works!")
                camera_index = i
                break
            cap.release()
    
    if cap is None or not cap.isOpened():
        print("\n✗ ERROR: No working camera found!")
        print("  - Check if camera is connected")
        print("  - Check if another app is using the camera")
        print("  - Try running as administrator")
        sys.exit(1)
    
    # Load face detector
    print("\nLoading face detector...")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    if face_cascade.empty():
        print("✗ ERROR: Could not load face detector!")
        sys.exit(1)
    
    print("✓ Face detector loaded")
    print("\n" + "-" * 50)
    print("Press 'q' to quit")
    print("-" * 50 + "\n")
    
    face_count = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Track max faces seen
        if len(faces) > face_count:
            face_count = len(faces)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Status text
        status = f"Faces detected: {len(faces)} | Frame: {frame_count}"
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow('Camera Test - Face Detection', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "=" * 50)
    print("  Test Complete!")
    print("=" * 50)
    print(f"  Camera index used: {camera_index}")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Max faces detected at once: {face_count}")
    print("\n  If face detection worked, you're ready for full setup!")
    print("=" * 50)


if __name__ == "__main__":
    test_camera()
