import cv2
import mediapipe as mp
import math
import time
import serial

# --- Set up Serial Communication with Arduino ---
try:
    # Change 'COM5' to your Arduino port (or '/dev/ttyXYZ' on macOS/Linux)
    ser = serial.Serial('COM5', 9600, timeout=1)
    time.sleep(2)  # Allow time for the connection to establish
except serial.SerialException:
    ser = None

# --- Mapping function: maps one range to another ---
def map_range(value, in_min, in_max, out_min, out_max):
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

# --- Function to compute Euclidean distance between two normalized points ---
def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# --- Initialize MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Optional: For drawing landmarks (for visualization)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- Variables for sleep detection ---
SLEEP_THRESHOLD = 5  # seconds
closed_start = None
sleepTriggered = False  # Flag to ensure "start" is sent only once

# --- Start capturing video from the webcam ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    h, w, _ = frame.shape

    # Flip the frame horizontally for a mirror view and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    # Default statuses and text messages
    status = "Can't detect"
    asleep_text = ""
    face_position_text = ""

    if results.multi_face_landmarks:
        # Process only the first detected face for simplicity
        face_landmarks = results.multi_face_landmarks[0]

        # Optionally draw facial landmarks on the frame
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )

        # --- Compute the Face Center (x, y) ---
        x_coords = [landmark.x for landmark in face_landmarks.landmark]
        y_coords = [landmark.y for landmark in face_landmarks.landmark]
        face_center_x = int(sum(x_coords) / len(x_coords) * w)
        face_center_y = int(sum(y_coords) / len(y_coords) * h)
        face_position_text = f"Face: x={face_center_x}, y={face_center_y}"

        # --- Calculate Eye Aspect Ratio (EAR) ---
        # Left eye landmarks (example indices)
        left_horizontal = euclidean_distance(face_landmarks.landmark[33],
                                               face_landmarks.landmark[133])
        left_vertical = euclidean_distance(face_landmarks.landmark[159],
                                             face_landmarks.landmark[145])
        left_EAR = left_vertical / left_horizontal

        # Right eye landmarks (example indices)
        right_horizontal = euclidean_distance(face_landmarks.landmark[362],
                                                face_landmarks.landmark[263])
        right_vertical = euclidean_distance(face_landmarks.landmark[386],
                                              face_landmarks.landmark[374])
        right_EAR = right_vertical / right_horizontal

        # Average EAR for both eyes
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Set a threshold: if average EAR is below this, assume eyes are closed.
        EAR_threshold = 0.4  # Adjust this value as needed

        if avg_EAR < EAR_threshold:
            status = "Eyes Closed"
            if closed_start is None:
                closed_start = time.time()
        else:
            status = "Eyes Open"
            closed_start = None
            sleepTriggered = False  # Reset the flag when eyes open

        # --- Map face center coordinates to servo angles ---
        # Map face center (x, y) from pixel range to servo angle range [0, 180]
        servo_x = map_range(face_center_x, 0, w, 0, 180)
        servo_y = map_range(face_center_y, 0, h, 0, 180)
        servo_y = 180 - servo_y  # Invert Y if needed
        # Format command for Arduino (e.g., "X90:Y120\n")
        command = f"X{servo_x}:Y{servo_y}\n"
        if ser:
            ser.write(command.encode())

    else:
        closed_start = None

    # Check if eyes have been closed continuously for more than the sleep threshold
    if closed_start is not None:
        elapsed = time.time() - closed_start
        if elapsed >= SLEEP_THRESHOLD:
            asleep_text = "Subject is asleep"
            # Send "start" command once if not already triggered
            if ser and not sleepTriggered:
                ser.write("start\n".encode())
                sleepTriggered = True

    # Display the status, sleep message, and face position on the frame
    cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if asleep_text:
        cv2.putText(frame, asleep_text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, face_position_text, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Face, Eye & Servo Control", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup: Release video capture, close windows, and close serial port if open
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
