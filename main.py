"""
Real-Time Face Recognition Based Attendance Monitoring System
Local Version - Works on Windows/Mac/Linux with standard webcam

Usage:
    python main.py

Controls:
    - Press 'q' to quit
    - Press 's' to manually capture and process current frame
    - Press 'r' to reset/reload known faces
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path

# Optional imports - will check availability
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("WARNING: face_recognition not installed. Face matching disabled.")

try:
    from tensorflow.keras.models import model_from_json
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not installed. Anti-spoofing disabled.")

try:
    import smtplib
    import ssl
    from email.message import EmailMessage
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    print("WARNING: Email modules not available.")

from pytz import timezone


# ============== CONFIGURATION ==============
class Config:
    """Configuration settings - EDIT THESE FOR YOUR SETUP"""
    
    # Paths (relative to this script)
    BASE_DIR = Path(__file__).parent.resolve()
    KNOWN_FACES_DIR = BASE_DIR / "face_recognition"
    MODELS_DIR = BASE_DIR / "Face_Antispoofing_System-main"
    ATTENDANCE_DIR = BASE_DIR / "Attendance folder"
    
    # Anti-spoofing model paths
    ANTISPOOFING_MODEL_JSON = MODELS_DIR / "antispoofing_models" / "antispoofing_model.json"
    ANTISPOOFING_MODEL_H5 = MODELS_DIR / "antispoofing_models" / "antispoofing_model.h5"
    HAAR_CASCADE = MODELS_DIR / "models" / "haarcascade_frontalface_default.xml"
    
    # Attendance files
    ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.txt"
    LOGIN_FILE = ATTENDANCE_DIR / "login.txt"
    
    # Face recognition settings
    FACE_RECOGNITION_TOLERANCE = 0.4  # Lower = stricter matching
    FACE_ENCODING_JITTERS = 10  # Higher = more accurate but slower
    
    # Camera settings
    CAMERA_INDEX = 0  # Change if you have multiple cameras
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # Anti-spoofing threshold
    SPOOF_THRESHOLD = 0.5  # Above this = spoof detected
    
    # Email settings (set these if you want email notifications)
    EMAIL_ENABLED = False  # Set to True to enable
    EMAIL_SENDER = "your_email@gmail.com"
    EMAIL_PASSWORD = "your_app_password"  # Use App Password, not regular password
    
    # Timezone
   Set timezone to Africa/Lusaka
    
    # User database (name -> [email, roll_number])
    USER_DATABASE = {
        'SRISRI': ['srisree322@gmail.com', '19R21A04K7'],
        'NAGASESHU': ['bnagaseshu2001@gmail.com', '19R21A04K2'],
        'VIKAS': ['vilasagaram.vikas@gmail.com', '20R25A0421'],
        'VAMSI': ['vamsi251002@gmail.com', '20R25A0420'],
    }


# ============== CORE SYSTEM ==============
class AttendanceSystem:
    def __init__(self, config=Config):
        self.config = config
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_cascade = None
        self.antispoofing_model = None
        
        self._setup_directories()
        self._load_face_cascade()
        self._load_antispoofing_model()
        self._load_known_faces()
        
    def _setup_directories(self):
        """Ensure required directories exist"""
        self.config.ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
        self.config.KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create attendance files if they don't exist
        if not self.config.ATTENDANCE_FILE.exists():
            with open(self.config.ATTENDANCE_FILE, 'w') as f:
                f.write("|| Attendance Log ||\n")
                f.write("=" * 50 + "\n")
        
        if not self.config.LOGIN_FILE.exists():
            with open(self.config.LOGIN_FILE, 'w') as f:
                f.write("")
    
    def _load_face_cascade(self):
        """Load Haar Cascade for face detection"""
        cascade_path = str(self.config.HAAR_CASCADE)
        
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            # Fall back to OpenCV's built-in cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        if self.face_cascade.empty():
            print("ERROR: Could not load face cascade classifier!")
            sys.exit(1)
        else:
            print("✓ Face detection model loaded")
    
    def _load_antispoofing_model(self):
        """Load the anti-spoofing deep learning model"""
        if not TENSORFLOW_AVAILABLE:
            print("✗ Anti-spoofing disabled (TensorFlow not installed)")
            return
            
        try:
            json_path = str(self.config.ANTISPOOFING_MODEL_JSON)
            h5_path = str(self.config.ANTISPOOFING_MODEL_H5)
            
            if os.path.exists(json_path) and os.path.exists(h5_path):
                with open(json_path, 'r') as f:
                    model_json = f.read()
                self.antispoofing_model = model_from_json(model_json)
                self.antispoofing_model.load_weights(h5_path)
                print("✓ Anti-spoofing model loaded")
            else:
                print(f"✗ Anti-spoofing model files not found at:")
                print(f"  JSON: {json_path}")
                print(f"  H5: {h5_path}")
        except Exception as e:
            print(f"✗ Error loading anti-spoofing model: {e}")
    
    def _load_known_faces(self):
        """Load and encode all known faces from the database"""
        if not FACE_RECOGNITION_AVAILABLE:
            print("✗ Face recognition disabled (face_recognition not installed)")
            return
            
        faces_dir = self.config.KNOWN_FACES_DIR
        if not faces_dir.exists():
            print(f"✗ Known faces directory not found: {faces_dir}")
            return
        
        print(f"Loading known faces from: {faces_dir}")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        for filename in os.listdir(faces_dir):
            filepath = faces_dir / filename
            
            if filepath.suffix.lower() not in valid_extensions:
                continue
                
            if filename.startswith('.'):
                continue
            
            try:
                image = face_recognition.load_image_file(str(filepath))
                encodings = face_recognition.face_encodings(
                    image, 
                    num_jitters=self.config.FACE_ENCODING_JITTERS,
                    model="large"
                )
                
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    name = filepath.stem.upper()  # Filename without extension
                    self.known_face_names.append(name)
                    print(f"  ✓ Loaded: {name}")
                else:
                    print(f"  ✗ No face found in: {filename}")
                    
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
        
        print(f"✓ Loaded {len(self.known_face_names)} known faces")
    
    def detect_faces(self, frame):
        """Detect faces in a frame using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def check_spoof(self, face_region):
        """Check if a face is real or spoofed"""
        if self.antispoofing_model is None:
            return False, 0.0  # Assume real if no model
        
        try:
            # Preprocess face for the model
            resized = cv2.resize(face_region, (160, 160))
            resized = resized.astype("float") / 255.0
            resized = np.expand_dims(resized, axis=0)
            
            # Get prediction
            prediction = self.antispoofing_model.predict(resized, verbose=0)[0][0]
            is_spoof = prediction > self.config.SPOOF_THRESHOLD
            
            return is_spoof, float(prediction)
        except Exception as e:
            print(f"Spoof check error: {e}")
            return False, 0.0
    
    def recognize_face(self, frame, face_location_dlib):
        """Recognize a face against known faces"""
        if not FACE_RECOGNITION_AVAILABLE or not self.known_face_encodings:
            return None, 0.0
        
        try:
            # Get face encoding
            encodings = face_recognition.face_encodings(
                frame, 
                [face_location_dlib],
                num_jitters=self.config.FACE_ENCODING_JITTERS,
                model="large"
            )
            
            if not encodings:
                return None, 0.0
            
            face_encoding = encodings[0]
            
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding,
                tolerance=self.config.FACE_RECOGNITION_TOLERANCE
            )
            
            # Get face distances for confidence
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, 
                face_encoding
            )
            
            if True in matches:
                best_match_idx = np.argmin(face_distances)
                if matches[best_match_idx]:
                    name = self.known_face_names[best_match_idx]
                    confidence = 1 - face_distances[best_match_idx]
                    return name, confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return None, 0.0
    
    def update_login_status(self, name):
        """Toggle login/logout status for a user"""
        login_file = self.config.LOGIN_FILE
        
        try:
            # Read current logins
            with open(login_file, 'r') as f:
                logged_in = [line.strip() for line in f.readlines() if line.strip()]
            
            is_logout = name in logged_in
            
            if is_logout:
                # Remove from logged in list
                logged_in.remove(name)
            else:
                # Add to logged in list
                logged_in.append(name)
            
            # Write back
            with open(login_file, 'w') as f:
                for user in logged_in:
                    f.write(user + '\n')
            
            return is_logout
            
        except Exception as e:
            print(f"Error updating login status: {e}")
            return False
    
    def mark_attendance(self, name):
        """Mark attendance for a recognized person"""
        now = datetime.now(timezone(self.config.TIMEZONE))
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%d/%m/%Y")
        
        is_logout = self.update_login_status(name)
        action = "logout" if is_logout else "login"
        
        # Write to attendance file
        try:
            with open(self.config.ATTENDANCE_FILE, 'a') as f:
                f.write(f"{name} {action} marked on {current_date} at {current_time}\n")
            print(f"✓ Attendance marked: {name} - {action} at {current_time}")
        except Exception as e:
            print(f"Error writing attendance: {e}")
        
        # Send email notification if enabled
        if self.config.EMAIL_ENABLED and EMAIL_AVAILABLE:
            self._send_notification(name, action, current_date, current_time)
        
        return action
    
    def _send_notification(self, name, action, date, time):
        """Send email notification"""
        if name not in self.config.USER_DATABASE:
            print(f"No email configured for {name}")
            return
        
        try:
            user_info = self.config.USER_DATABASE[name]
            email_receiver = user_info[0]
            roll_number = user_info[1]
            
            subject = 'Attendance Notification'
            body = f"""
            Attendance Update
            -----------------
            Name: {name}
            Roll No: {roll_number}
            Action: {action.upper()}
            Date: {date}
            Time: {time}
            
            This is an automated message from the Face Recognition Attendance System.
            """
            
            em = EmailMessage()
            em['From'] = self.config.EMAIL_SENDER
            em['To'] = email_receiver
            em['Subject'] = subject
            em.set_content(body)
            
            context = ssl.create_default_context()
            
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                smtp.login(self.config.EMAIL_SENDER, self.config.EMAIL_PASSWORD)
                smtp.sendmail(self.config.EMAIL_SENDER, email_receiver, em.as_string())
            
            print(f"✓ Email sent to {email_receiver}")
            
        except Exception as e:
            print(f"Email error: {e}")


# ============== MAIN APPLICATION ==============
def run_attendance_system():
    """Main function to run the attendance system"""
    
    print("\n" + "=" * 60)
    print("  Real-Time Face Recognition Attendance System")
    print("  Local Version")
    print("=" * 60 + "\n")
    
    # Initialize system
    system = AttendanceSystem()
    
    # Initialize camera
    print(f"\nOpening camera {Config.CAMERA_INDEX}...")
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        print("Try changing CAMERA_INDEX in Config class")
        sys.exit(1)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    
    print("✓ Camera opened successfully")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Capture and process frame")
    print("  'r' - Reload known faces")
    print("\n" + "-" * 60)
    
    # Cooldown to prevent multiple detections
    last_detection_time = {}
    DETECTION_COOLDOWN = 5  # seconds
    
    frame_count = 0
    process_every_n_frames = 5  # Process every 5th frame for performance
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Detect faces
        faces = system.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            color = (255, 255, 0)  # Yellow - detecting
            label = "Detecting..."
            
            # Only process every N frames for performance
            if frame_count % process_every_n_frames == 0:
                # Extract face region with padding
                padding = 5
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                face_region = frame[y1:y2, x1:x2]
                
                # Check for spoofing
                is_spoof, spoof_score = system.check_spoof(face_region)
                
                if is_spoof:
                    color = (0, 0, 255)  # Red
                    label = f"SPOOF ({spoof_score:.2f})"
                else:
                    # Convert OpenCV rect to dlib format (top, right, bottom, left)
                    face_location_dlib = (y, x + w, y + h, x)
                    
                    # Recognize face
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    name, confidence = system.recognize_face(rgb_frame, face_location_dlib)
                    
                    if name:
                        # Check cooldown
                        current_time = datetime.now().timestamp()
                        if name not in last_detection_time or \
                           (current_time - last_detection_time[name]) > DETECTION_COOLDOWN:
                            
                            # Mark attendance
                            action = system.mark_attendance(name)
                            last_detection_time[name] = current_time
                        
                        color = (0, 255, 0)  # Green
                        label = f"{name} ({confidence:.0%})"
                    else:
                        color = (0, 165, 255)  # Orange
                        label = "Unknown"
            
            # Draw on frame
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show status bar
        status = f"Faces: {len(faces)} | Press 'q' to quit"
        cv2.putText(display_frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Face Recognition Attendance System', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            print("\nManual capture triggered")
            cv2.imwrite('captured_frame.jpg', frame)
            print("Frame saved to captured_frame.jpg")
        elif key == ord('r'):
            print("\nReloading known faces...")
            system._load_known_faces()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ System shutdown complete")


if __name__ == "__main__":
    run_attendance_system()
