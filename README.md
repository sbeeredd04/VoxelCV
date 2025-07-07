# VoxelCV

VoxelCV is an advanced computer vision toolkit for human posture analysis and fall detection, leveraging pose estimation and AI-powered image analysis. It is designed for applications in health monitoring, activity recognition, and safety surveillance, especially where fall detection is critical (e.g., elderly care, workplaces).

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage Guide](#usage-guide)
  - [Posture Analysis](#posture-analysis)
  - [Real-Time Posture Classification](#real-time-posture-classification)
  - [Fall Detection (AI-Powered)](#fall-detection-ai-powered)
- [Sample Code](#sample-code)
- [Requirements](#requirements)
- [Directory Structure](#directory-structure)
- [License & Credits](#license--credits)
- [Contact](#contact)

---

## Features

- **Human Pose Estimation**: Detects and visualizes 33 body landmarks using MediaPipe Pose.
- **Biomechanical Angle Analysis**: Calculates and displays neck angles and other posture metrics.
- **Batch & Real-Time Processing**: Supports both static image datasets and live webcam feeds.
- **AI-Powered Fall Detection**: Uses Google Gemini Vision API for robust fall classification and explanation.
- **Rich Visualization**: Annotates images with landmarks, angle overlays, and color-coded fall status.
- **Flexible Output**: View results in OpenCV windows or Matplotlib for Jupyter/Colab compatibility.

---

## Architecture

Below is a high-level architecture of the VoxelCV project:

```mermaid
flowchart TD
    %% Data Flow
    A[Image Dataset] --> B[Posture Analysis (posture.ipynb)]
    B --> C[MediaPipe Pose Estimation]
    C --> D[Landmark Extraction & Angle Calculation]
    D --> E[Visualization (OpenCV/Matplotlib)]

    A --> F[Fall Detection (falltest.py)]
    F --> G[Google Gemini Vision API]
    G --> H[Fall Classification (NORMAL/FALLING/FALLEN)]
    H --> I[Annotated Output with Explanation]

    %% Core Libraries
    subgraph Libraries
        J(OpenCV)
        K(MediaPipe)
        L(NumPy)
        M(Matplotlib)
        N(Pillow)
    end

    B --> J
    B --> K
    B --> L
    B --> M
    F --> J
    F --> N
    F --> G
```

---

## Getting Started

### Prerequisites

- Python 3.7+
- [Google Gemini API access & key](https://ai.google.dev/gemini-api/)
- Required Python packages (see [Requirements](#requirements))

### Installation

```bash
pip install opencv-python mediapipe numpy matplotlib pillow google-generativeai
```

---

## Usage Guide

### Posture Analysis

- **File:** `posture.ipynb`
- **Function:** Batch processes images to detect human pose, compute biomechanical angles, and visualize results.
- **Workflow:**
  1. Randomly selects PNG images from `dataset/newDataset/`.
  2. Runs MediaPipe Pose to detect body landmarks.
  3. Calculates key angles (e.g., neck).
  4. Annotates images with landmarks, connecting lines, and angle values.
  5. Displays annotated images via OpenCV or Matplotlib.

### Real-Time Posture Classification

- **File:** `postureCalssifier.py`
- **Function:** Processes live webcam streams for posture visualization and angle measurement.
- **Features:** 
  - Real-time landmark detection
  - Drawing joint connections
  - Graceful exit via keyboard

### Fall Detection (AI-Powered)

- **File:** `falltest.py`
- **Function:** Uses Google Gemini Vision API for image-based fall classification and textual explanation.
- **Workflow:**
  1. Loads an image for analysis.
  2. Sends image + prompt to Gemini API.
  3. Receives category: **NORMAL**, **FALLING**, or **FALLEN**.
  4. Overlays result and explanation on the image.
  5. Color-codes output (green/yellow/red) and displays annotated image.

---

## Sample Code

**Pose Detection & Angle Calculation**
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
    # [Extract coordinates, calculate angles...]
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Annotated', image)
    cv2.waitKey(0)
```

**Fall Detection with Gemini API**
```python
from google import genai
import PIL.Image

genai.configure(api_key="YOUR_API_KEY")
client = genai.GenerativeModel('gemini-pro-vision')
prompt = "Analyze this image for potential fall detection. Classify as NORMAL, FALLING, or FALLEN. Explain your reasoning."
image = PIL.Image.open('sample.png')
response = client.generate_content([prompt, image])
print(response.text)
```

---

## Requirements

- Python 3.7 or newer
- OpenCV (`opencv-python`)
- MediaPipe
- NumPy
- Matplotlib
- Pillow (PIL)
- `google-generativeai` (for Gemini Vision API)

Install with:
```bash
pip install opencv-python mediapipe numpy matplotlib pillow google-generativeai
```

---

## Directory Structure

```
VoxelCV/
├── posture.ipynb              # Notebook for batch posture analysis
├── postureCalssifier.py       # Real-time webcam posture classifier
├── falltest.py                # AI-based fall detection script
├── dataset/
│   └── newDataset/            # Input images (.png)
└── README.md                  # Project documentation
```

---

## License & Credits

- **License:** For research and educational use. For commercial or clinical use, contact the author.
- **Credits:**
  - [MediaPipe](https://mediapipe.dev/)
  - [OpenCV](https://opencv.org/)
  - [Google Gemini](https://ai.google.dev/gemini-api/)

---

## Contact

For questions or suggestions, please open an issue on this repository.

---

> ![image1](image1)
*Screenshot: Example of architecture section error and mermaid diagram fix.*
