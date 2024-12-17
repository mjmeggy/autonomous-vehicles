import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe modules for Pose and Face Mesh detection
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Constants for eye-blink detection
LEFT_EYE_LANDMARKS = [159, 145]  # Top and bottom eyelid landmarks for left eye
RIGHT_EYE_LANDMARKS = [386, 374]  # Top and bottom eyelid landmarks for right eye
BLINK_THRESHOLD = 0.03  # Vertical eye aspect ratio (EAR) threshold for blinking

# Thresholds for behavior detection
BLINK_FATIGUE_THRESHOLD = 20  # Blinks in a short period indicating fatigue
POSTURE_THRESHOLD = 0.3       # Threshold for significant head tilt

# Function to calculate eye aspect ratio
def calculate_ear(landmarks, eye_indices):
    top = landmarks[eye_indices[0]].y
    bottom = landmarks[eye_indices[1]].y
    return abs(top - bottom)

# Function to calculate head tilt (pose)
def calculate_head_tilt(landmarks):
    nose_tip = landmarks[0]  # Nose tip landmark
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    # Calculate horizontal shoulder tilt
    shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
    return shoulder_tilt

# Initialize video capture and Mediapipe solutions
cap = cv2.VideoCapture(0)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Variables for blink detection and behavior logic
blink_count = 0
last_blink_time = 0
start_time = time.time()
behavior_status = "Normal Behavior"

print("Press 'q' to exit the program.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Process Pose and Face Mesh
    pose_results = pose.process(image_rgb)
    face_results = face_mesh.process(image_rgb)

    # Re-enable frame writing for drawing
    image_rgb.flags.writeable = True
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Eye Blink Detection and Behavior Analysis
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Calculate EAR for both eyes
            left_ear = calculate_ear(landmarks, LEFT_EYE_LANDMARKS)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_LANDMARKS)

            # Check for blink
            if left_ear < BLINK_THRESHOLD and right_ear < BLINK_THRESHOLD:
                if time.time() - last_blink_time > 0.2:  # Prevent double counting
                    blink_count += 1
                    last_blink_time = time.time()

            # Display EAR and blink count
            cv2.putText(frame, f"Left EAR: {left_ear:.3f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink Count: {blink_count}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Posture Analysis (Distraction/Unusual Behavior)
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        head_tilt = calculate_head_tilt(landmarks)

        # Behavior Detection Logic
        if head_tilt > POSTURE_THRESHOLD:
            behavior_status = "Distraction Detected"
        elif blink_count > BLINK_FATIGUE_THRESHOLD:
            behavior_status = "Fatigue Detected"
        else:
            behavior_status = "Normal Behavior"

        # Reset blink count every 60 seconds
        if time.time() - start_time > 60:
            blink_count = 0
            start_time = time.time()

        # Draw Pose landmarks
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display Behavior Status
    cv2.putText(frame, f"Behavior: {behavior_status}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Behavior Detection with Body Language and Eye Blink", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
