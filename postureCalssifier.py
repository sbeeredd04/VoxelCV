import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(point1, point2, vertex):
    """Calculate angle between three points with error handling"""
    try:
        a = np.array(point1)
        b = np.array(vertex)
        c = np.array(point2)
        
        ba = a - b
        bc = c - b
        
        # Check for zero vectors
        if np.all(ba == 0) or np.all(bc == 0):
            return 0
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 0

def get_perpendicular_line(point1, point2, length=200):
    """Calculate perpendicular line with error handling"""
    try:
        # Get midpoint
        mid_x = (point1[0] + point2[0]) // 2
        mid_y = (point1[1] + point2[1]) // 2
        
        # Calculate direction vector
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        # Check for zero magnitude
        magnitude = math.sqrt(dx*dx + dy*dy)
        if magnitude < 1e-6:
            return (mid_x, mid_y), (mid_x, mid_y - length), (mid_x, mid_y + length)
        
        # Get perpendicular vector (-dy, dx)
        perp_dx = -dy
        perp_dy = dx
        
        # Normalize and scale
        perp_dx = int((perp_dx / magnitude) * length)
        perp_dy = int((perp_dy / magnitude) * length)
        
        perp_point1 = (mid_x + perp_dx, mid_y + perp_dy)
        perp_point2 = (mid_x - perp_dx, mid_y - perp_dy)
        
        return (mid_x, mid_y), perp_point1, perp_point2
    except Exception as e:
        print(f"Error calculating perpendicular line: {e}")
        return (point1[0], point1[1]), point1, point2

def add_legend(image, items, start_y=30):
    """Add color-coded legend to image with smaller font"""
    for i, (text, color) in enumerate(items):
        y = start_y + (i * 20)  # Reduced spacing
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 1, cv2.LINE_AA)  # Smaller font
        cv2.line(image, (160, y-5), (200, y-5), color, 2)  # Adjusted line position

def calculate_midpoint(point1, point2):
    """Calculate midpoint between two points"""
    return ((point1[0] + point2[0])//2, (point1[1] + point2[1])//2)

def get_reference_line(hip_point, length=200, is_perpendicular=False):
    """Calculate reference line from hip point"""
    if is_perpendicular:
        return (hip_point[0], hip_point[1] - length)  # Vertical line up
    else:
        return (hip_point[0], hip_point[1] + length)  # Vertical line down

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam initialized")
cv2.namedWindow('Posture Analysis', cv2.WINDOW_NORMAL)
print("Press 'q' to quit")

# Define colors
NECK_COLOR = (255, 0, 0)      # Blue
BODY_COLOR = (0, 255, 255)    # Yellow
SHOULDER_COLOR = (0, 255, 0)  # Green
LEG_COLOR = (255, 0, 255)     # Magenta
REFERENCE_COLOR = (128, 128, 128)  # Gray
BENDING_COLOR = (255, 165, 0)  # Orange

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape
        annotated_image = frame.copy()

        try:
            # Extract key landmarks using the provided indices
            nose = (int(landmarks[0].x * w), int(landmarks[0].y * h))
            left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
            right_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            left_hip = (int(landmarks[23].x * w), int(landmarks[23].y * h))
            right_hip = (int(landmarks[24].x * w), int(landmarks[24].y * h))
            left_knee = (int(landmarks[25].x * w), int(landmarks[25].y * h))
            right_knee = (int(landmarks[26].x * w), int(landmarks[26].y * h))

            # Calculate additional points (33: hip_mid, 34: neck_mid)
            hip_mid = calculate_midpoint(left_hip, right_hip)  # Point 33
            neck_mid = calculate_midpoint(left_shoulder, right_shoulder)  # Point 34

            # Calculate body angle using shoulders, hips, and knees
            knee_mid = calculate_midpoint(left_knee, right_knee)
            # Calculate angle between vertical line and body line
            vertical_ref = (hip_mid[0], hip_mid[1] - 100)  # Point 100 pixels above hip
            body_angle = calculate_angle(vertical_ref, hip_mid, knee_mid)

            # Determine if sitting/standing based on body angle
            # If angle between vertical and legs is small (< 10°), person is standing
            is_standing = body_angle < 10
            reference_point = get_reference_line(hip_mid, 
                                              is_perpendicular=not is_standing)

            # Calculate key angles
            # 1. Slouching angle (shoulder line vs reference)
            shoulder_vector = np.array([right_shoulder[0] - left_shoulder[0],
                                      right_shoulder[1] - left_shoulder[1]])
            reference_vector = np.array([reference_point[0] - hip_mid[0],
                                       reference_point[1] - hip_mid[1]])
            slouching_angle = calculate_angle(left_shoulder, right_shoulder, reference_point)

            # 2. Bending/Tilting angle (neck to vertical)
            vertical_neck_ref = (neck_mid[0], neck_mid[1] - 100)
            bending_angle = calculate_angle(vertical_neck_ref, neck_mid, nose)

            # 3. Neck angle (nose to neck to hip)
            neck_angle = calculate_angle(nose, neck_mid, hip_mid)

            # Draw visualization lines
            # Body structure
            cv2.line(annotated_image, left_shoulder, right_shoulder, SHOULDER_COLOR, 2)
            cv2.line(annotated_image, left_hip, right_hip, BODY_COLOR, 2)
            cv2.line(annotated_image, hip_mid, neck_mid, BODY_COLOR, 2)
            cv2.line(annotated_image, hip_mid, knee_mid, LEG_COLOR, 2)
            cv2.line(annotated_image, hip_mid, vertical_ref, REFERENCE_COLOR, 1, cv2.LINE_AA)
            cv2.line(annotated_image, neck_mid, vertical_neck_ref, REFERENCE_COLOR, 1, cv2.LINE_AA)
            cv2.line(annotated_image, neck_mid, nose, NECK_COLOR, 2)

            # Draw key points
            cv2.circle(annotated_image, hip_mid, 4, (0, 0, 255), -1)  # Point 33
            cv2.circle(annotated_image, neck_mid, 4, (0, 255, 0), -1)  # Point 34
            cv2.circle(annotated_image, knee_mid, 4, (255, 0, 255), -1)  # Knee midpoint

            # Add measurements with improved status indicators
            measurements = [
                (f"Body Angle: {int(body_angle)}° {'(Standing)' if is_standing else '(Sitting)'}", 
                 BODY_COLOR),
                (f"Slouching: {int(slouching_angle)}° {'(Good)' if slouching_angle < 10 else '(Poor)'}", 
                 SHOULDER_COLOR),
                (f"Bending: {int(bending_angle)}° {'(Good)' if bending_angle < 15 else '(Poor)'}", 
                 BENDING_COLOR),
                (f"Neck: {int(neck_angle)}° {'(Good)' if 80 < neck_angle < 100 else '(Poor)'}",
                 NECK_COLOR)
            ]
            add_legend(annotated_image, measurements, start_y=30)

            # Add posture feedback based on standing/sitting
            feedback = []
            if is_standing:
                if body_angle > 5:
                    feedback.append("Stand straighter - Align with vertical")
                if slouching_angle > 10:
                    feedback.append("Level your shoulders")
                if bending_angle > 15:
                    feedback.append("Reduce forward head posture")
            else:  # sitting
                if slouching_angle > 10:
                    feedback.append("Sit up straight - Fix slouching")
                if bending_angle > 20:
                    feedback.append("Reduce forward bend")
                if not (80 < neck_angle < 100):
                    feedback.append("Adjust neck position")

            # Add reference lines legend
            reference_legend = [
                ("Vertical Reference", REFERENCE_COLOR),
                ("Body Line", BODY_COLOR),
                ("Shoulder Line", SHOULDER_COLOR),
                ("Neck Line", NECK_COLOR),
                ("Leg Line", LEG_COLOR)
            ]
            add_legend(annotated_image, reference_legend, start_y=150)

            if feedback:
                y_pos = 250
                for msg in feedback:
                    cv2.putText(annotated_image, f"! {msg}", (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    y_pos += 20

            # Draw landmark connections
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1, circle_radius=1)
            )

        except Exception as e:
            print(f"Error processing landmarks: {e}")
            continue

        cv2.imshow('Posture Analysis', annotated_image)
    else:
        cv2.imshow('Posture Analysis', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Application closed")