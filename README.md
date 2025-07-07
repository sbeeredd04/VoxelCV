# VoxelCV

VoxelCV is a computer vision toolkit focused on analyzing human posture and detecting falls using advanced pose estimation and AI-based image analysis. Its main applications include health monitoring, activity recognition, and safety surveillance, particularly in environments where fall detection is critical (elderly care, workplaces, etc.).

This project leverages the following technologies:
- **Python** – main programming language
- **OpenCV** – image processing and visualization
- **MediaPipe** – state-of-the-art pose landmark detection
- **Matplotlib** – visualization of annotated images
- **NumPy** – mathematical operations and array handling
- **Google Gemini API** – AI-powered vision model for fall classification
- **PIL (Pillow)** – image IO for AI analysis

The repository contains two major functionalities:
- **Posture Analysis**: Visualize and analyze human posture in images using pose landmarks.
- **Fall Detection**: Classify and explain fall events in images using Gemini's vision AI.

---

## Features

### 1. Posture Analysis
- **Batch processing**: Randomly selects and processes images from a dataset.
- **Pose estimation**: Detects 33 body landmarks per person using MediaPipe.
- **Angle measurement**: Computes biomechanical angles (e.g., neck angle) from detected landmarks.
- **Landmark drawing**: Annotates images with joints, neck, nose, and hips, and overlays calculated angles.
- **Visualization**: Outputs can be displayed using OpenCV (real-time window) or Matplotlib (static notebook visualization).

### 2. Fall Detection
- **AI-based analysis**: Uses Google Gemini (via Python SDK) to classify posture in an image as NORMAL, FALLING, or FALLEN.
- **Explanations**: AI provides a textual explanation for its classification.
- **Visual feedback**: Annotates images with results, using color-coded overlays (green/yellow/red) based on the fall status.
- **Timestamping**: Each analysis is timestamped for tracking.

---

## Project Architecture

```mermaid
flowchart TD
    A[Image Dataset] --> B[Posture Analysis (posture.ipynb)]
    B --> C[MediaPipe Pose Estimation]
    C --> D[Landmark Extraction & Angle Calculation]
    D --> E[Visualization (OpenCV/Matplotlib)]
    A --> F[Fall Detection (falltest.py)]
    F --> G[Google Gemini Vision API]
    G --> H[Fall Classification (NORMAL/FALLING/FALLEN)]
    H --> I[Annotated Output with Explanation]
    subgraph Core Libraries
        J(OpenCV)
        K(MediaPipe)
        L(NumPy)
        M(Matplotlib)
        N(PIL)
    end
    B & F --> J
    B & F --> L
    B --> C
    B --> M
    F --> N
    F --> G
```

### Breakdown of Components

#### 1. Posture Analysis (`posture.ipynb`)
- **Image selection**: Randomly selects 10 PNG images from a dataset.
- **Pose detection**: Uses MediaPipe Pose to find human joints.
- **Landmark processing**: Extracts coordinates for key points (nose, shoulders, hips).
- **Angle calculation**: Computes the angle at the neck using vector math.
- **Annotation**: Draws landmarks, neck, nose, hip midpoints, and connecting lines. Displays the calculated angle on the image.
- **Output**: Shows annotated images either in a GUI window or inline (Jupyter/IPython).

#### 2. Posture Classifier (`postureCalssifier.py`)
- **Real-time webcam support**: Can process video streams for live posture recognition.
- **Landmark connection drawing**: Visualizes detected joints and their connections.
- **Application control**: Closes gracefully on user command.

#### 3. Fall Detection (`falltest.py`)
- **Image input**: Loads an image for fall analysis.
- **AI-powered inference**: Sends the image to Google Gemini’s generative vision model with a specific prompt for fall detection.
- **Result parsing**: Receives and parses the category (NORMAL, FALLING, FALLEN) and explanation.
- **Annotation**: Overlays results on the image with contextual coloring.
- **Display**: Shows before/after images and prints the analysis.

---

## Example Workflow

1. **Prepare Dataset**: Place PNG images of people in `dataset/newDataset/`.
2. **Run Posture Analysis**: Execute `posture.ipynb` to batch-process and visualize pose landmarks and neck angles.
3. **Fall Detection**: Use `falltest.py` to analyze individual images for falls, leveraging the Gemini API for AI-powered explanations.

---

## Sample Code Snippet

**Pose Detection and Angle Calculation**
```python
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

image = cv2.imread('sample.png')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(rgb_image)
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    # Example: Calculate neck angle using coordinates
    # [Coordinate extraction logic...]
    # [Angle calculation logic...]
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Annotated', image)
    cv2.waitKey(0)
```

**Fall Detection with Gemini**
```python
from google import genai
import PIL.Image

genai.configure(api_key="YOUR_API_KEY")
client = genai.GenerativeModel('gemini-pro-vision')
prompt = "Analyze this image for potential fall detection..."
image = PIL.Image.open('sample.png')
response = client.generate_content([prompt, image])
print(response.text)
```

---

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- Pillow (PIL)
- google-genai (for Gemini vision API access)

Install dependencies:
```bash
pip install opencv-python mediapipe numpy matplotlib pillow google-generativeai
```

---

## Directory Structure

```
VoxelCV/
├── posture.ipynb              # Jupyter notebook for posture analysis
├── falltest.py                # Script for fall detection using Gemini
├── postureCalssifier.py       # Real-time posture classifier with visualization
├── dataset/
│   └── newDataset/            # Folder with input images (.png)
└── README.md                  # This file
```

---

## License

This project is for research and educational use. For commercial or clinical use, consult the author.

---

## Credits

- MediaPipe: [https://mediapipe.dev/](https://mediapipe.dev/)
- OpenCV: [https://opencv.org/](https://opencv.org/)
- Google Gemini: [https://ai.google.dev/gemini-api/](https://ai.google.dev/gemini-api/)

---

## Contact

For questions, open an issue on this repository.

```
