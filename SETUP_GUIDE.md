# Local Setup Guide: Face Recognition Attendance System

## Quick Start (TL;DR)

```bash
# 1. Create project folder and copy files
# 2. Install Python 3.8-3.10
# 3. Install dependencies:
pip install -r requirements.txt
# 4. Run:
python main.py
```

---

## Detailed Setup Instructions

### Step 1: System Requirements

| Requirement | Recommended |
|-------------|-------------|
| **OS** | Windows 10/11, macOS 10.15+, Ubuntu 20.04+ |
| **Python** | 3.8, 3.9, or 3.10 (NOT 3.11+ due to dlib issues) |
| **RAM** | 8GB minimum, 16GB recommended |
| **Webcam** | Any USB or built-in camera |
| **Storage** | ~2GB for dependencies |

### Step 2: Install Python

**Check your Python version:**
```bash
python --version
# or
python3 --version
```

If you need Python 3.10:
- **Windows**: Download from https://www.python.org/downloads/release/python-31011/
- **macOS**: `brew install python@3.10`
- **Ubuntu**: `sudo apt install python3.10 python3.10-venv python3.10-dev`

### Step 3: Install Build Tools (Required for dlib/face_recognition)

#### Windows:
1. Install Visual Studio Build Tools:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Select "Desktop development with C++"
   - Install and restart

2. Install CMake:
   - Download from: https://cmake.org/download/
   - Add to PATH during installation

#### macOS:
```bash
xcode-select --install
brew install cmake
```

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install build-essential cmake
sudo apt install libopenblas-dev liblapack-dev
sudo apt install libx11-dev libgtk-3-dev
```

### Step 4: Set Up Project Directory

```bash
# Create project folder
mkdir face-attendance-system
cd face-attendance-system

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 5: Copy Project Files

Your folder structure should look like this:

```
face-attendance-system/
├── main.py                          # Main application (from this setup)
├── requirements.txt                 # Dependencies
├── Face_Antispoofing_System-main/   # From original project
│   ├── antispoofing_models/
│   │   ├── antispoofing_model.h5
│   │   └── antispoofing_model.json
│   └── models/
│       └── haarcascade_frontalface_default.xml
├── face_recognition/                # From original project (your known faces)
│   ├── Person1.jpg
│   ├── Person2.jpg
│   └── ...
└── Attendance folder/               # Will be created automatically
    ├── attendance.txt
    └── login.txt
```

### Step 6: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install main dependencies
pip install -r requirements.txt
```

#### If dlib installation fails:

**Option A: Install pre-built wheel (Windows)**
```bash
# For Python 3.10 on Windows:
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl
```

**Option B: Use conda (All platforms)**
```bash
conda install -c conda-forge dlib
pip install face_recognition
```

**Option C: Build from source**
```bash
pip install cmake
pip install dlib --verbose
```

### Step 7: Add Your Known Faces

1. Create the `face_recognition` folder if it doesn't exist
2. Add clear, front-facing photos of each person
3. **Name each file with the person's name** (e.g., `JOHN_DOE.jpg`)
4. Supported formats: `.jpg`, `.jpeg`, `.png`

**Tips for good face images:**
- Clear, well-lit photos
- Front-facing (looking at camera)
- One person per image
- Minimum 200x200 pixels

### Step 8: Configure the System

Edit `main.py` and modify the `Config` class:

```python
class Config:
    # Camera (0 = default, try 1 or 2 if wrong camera)
    CAMERA_INDEX = 0
    
    # Face matching strictness (lower = stricter)
    FACE_RECOGNITION_TOLERANCE = 0.4
    
    # Email notifications (optional)
    EMAIL_ENABLED = False  # Set True to enable
    EMAIL_SENDER = "your_email@gmail.com"
    EMAIL_PASSWORD = "your_app_password"
    
    # User database
    USER_DATABASE = {
        'JOHN_DOE': ['john@email.com', 'EMP001'],
        'JANE_SMITH': ['jane@email.com', 'EMP002'],
    }
```

### Step 9: Run the System

```bash
python main.py
```

**Controls:**
- `q` - Quit the application
- `s` - Manually capture current frame
- `r` - Reload known faces (if you add new ones)

---

## Troubleshooting

### "No module named 'face_recognition'"
```bash
pip install face_recognition
# If that fails, install dlib first:
pip install dlib
pip install face_recognition
```

### "Camera not found" / Black screen
```python
# Try different camera index in Config:
CAMERA_INDEX = 1  # or 2, 3, etc.
```

### "No faces found in known images"
- Ensure images are clear and front-facing
- Check file format (use .jpg or .png)
- Verify faces are clearly visible

### Slow performance
```python
# In main.py, increase this value:
process_every_n_frames = 10  # Process fewer frames
```

### TensorFlow warnings
These are usually harmless. To suppress:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

### "Could not load anti-spoofing model"
- Check that `antispoofing_model.h5` and `.json` exist
- System will still work (just without spoof detection)

---

## Testing Without Full Setup

If you just want to test face detection without face_recognition:

```python
# test_camera.py - Quick camera test
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Face Detection Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run with just:
```bash
pip install opencv-python
python test_camera.py
```

---

## Gmail App Password Setup (For Email Notifications)

1. Go to https://myaccount.google.com/security
2. Enable 2-Factor Authentication
3. Go to "App passwords"
4. Create new app password for "Mail"
5. Use this 16-character password in Config.EMAIL_PASSWORD

---

## Next Steps

Once basic setup works:
1. Add more known faces
2. Adjust tolerance for accuracy
3. Enable email notifications
4. Consider adding a database for better logging
