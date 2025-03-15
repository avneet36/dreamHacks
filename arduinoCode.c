import cv2       
import mediapipe as mp
import serial
import time

# Initialize Serial Communication (Change 'COM3' to match your system)
try:
    ser = serial.Serial('/dev/cu.usbserial-A5069RR4', 9600, timeout=1)
    time.sleep(2)  # Allow time for the connection to establish
    print("Connected to Arduino")
except serial.SerialException:
    print("Failed to connect to Arduino!")
    ser = None

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# Open the webcam
cap = cv2.VideoCapture(0)

# Servo control variables
servo_x, servo_y = 90, 90  # Start servos in the center
MOVEMENT_STEP = 3  # Larger step size for faster movement
TOLERANCE = 1  # Allowable margin of error

# **Camera to Laser Correction Factors (Adjust these manually)**
X_CORRECTION = 10  # Shift left/right
Y_CORRECTION = -15   # Shift up/down

# Map function: Maps screen coordinates (0  -640) to servo angles (0-180)
def map_range(value, in_min, in_max, out_min, out_max):
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural movement
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert frame to RGB (needed for Mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get eye landmarks from detection
            keypoints = detection.location_data.relative_keypoints
            left_eye = keypoints[0]  # Left eye
            right_eye = keypoints[1]  # Right eye

            # Convert normalized coordinates (0-1) to pixels
            left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
            right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

            # Determine which eye is closer (lower Y = closer)
            closer_eye = (left_eye_x, left_eye_y) if left_eye_y < right_eye_y else (right_eye_x, right_eye_y)

            # **Apply perspective correction**
            corrected_x = closer_eye[0] + X_CORRECTION
            corrected_y = closer_eye[1] + Y_CORRECTION

            # Convert eye position to servo angles (0-180 degrees)
            target_x = map_range(corrected_x, 0, w, 0, 180)
            target_y = map_range(corrected_y, 0, h, 0, 180)

            # Send servo positions to Arduino
            if ser:
                command = f'X{target_x}:Y{target_y}\n'
                ser.write(command.encode())
                print(f"Sent to Arduino: {command}")

            # Draw tracking circles on eyes
            cv2.circle(frame, (left_eye_x, left_eye_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (right_eye_x, right_eye_y), 5, (255, 0, 0), -1)
            cv2.circle(frame, closer_eye, 8, (0, 0, 255), -1)  # Red dot on tracked eye

    # Display the output
    cv2.imshow("Eye Tracking with Perspective Correction", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
