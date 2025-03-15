import cv2
import mediapipe as mp
import math
import time
import json  # For JSON response

# --- Set up Serial Communication with Arduino ---
# Add serial communication setup here if required...

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
SLEEP_THRESHOLD = 5  # seconds for detecting sleep after eyes are closed
closed_start = None

def detect_eye_sleep_status(frame):
    global closed_start
    
    # Get frame dimensions
    h, w, _ = frame.shape

    # Flip the frame horizontally for a mirror view and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    # Default statuses and text messages
    status = "Can't detect"
    asleep_text = "No"
    face_position_text = ""

    # Variables for face and eye positions
    face_position = {"top": 0, "left": 0, "width": 0, "height": 0}
    eyes = [{"top": 0, "left": 0, "width": 0, "height": 0}, {"top": 0, "left": 0, "width": 0, "height": 0}]

    if results.multi_face_landmarks:
        # Process only the first detected face for simplicity
        face_landmarks = results.multi_face_landmarks[0]

        # --- Compute the Face Center (x, y) ---
        x_coords = [landmark.x for landmark in face_landmarks.landmark]
        y_coords = [landmark.y for landmark in face_landmarks.landmark]
        face_center_x = int(sum(x_coords) / len(x_coords) * w)
        face_center_y = int(sum(y_coords) / len(y_coords) * h)
        face_position_text = f"Face: x={face_center_x}, y={face_center_y}"

        # --- Get face bounding box (top, left, width, height) ---
        face_position["left"] = int(min(x_coords) * w)
        face_position["top"] = int(min(y_coords) * h)
        face_position["width"] = int((max(x_coords) - min(x_coords)) * w)
        face_position["height"] = int((max(y_coords) - min(y_coords)) * h)

        # --- Calculate Eye Aspect Ratio (EAR) ---
        # Left eye landmarks (example indices)
        left_horizontal = euclidean_distance(face_landmarks.landmark[33], face_landmarks.landmark[133])
        left_vertical = euclidean_distance(face_landmarks.landmark[159], face_landmarks.landmark[145])
        left_EAR = left_vertical / left_horizontal

        # Right eye landmarks (example indices)
        right_horizontal = euclidean_distance(face_landmarks.landmark[362], face_landmarks.landmark[263])
        right_vertical = euclidean_distance(face_landmarks.landmark[386], face_landmarks.landmark[374])
        right_EAR = right_vertical / right_horizontal

        # --- Get Eye Bounding Boxes ---
        left_eye_x = int(face_landmarks.landmark[33].x * w)
        left_eye_y = int(face_landmarks.landmark[33].y * h)
        right_eye_x = int(face_landmarks.landmark[362].x * w)
        right_eye_y = int(face_landmarks.landmark[362].y * h)

        eyes[0] = {"top": left_eye_y - 15, "left": left_eye_x - 15, "width": 30, "height": 30}
        eyes[1] = {"top": right_eye_y - 15, "left": right_eye_x - 15, "width": 30, "height": 30}

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
            closed_start = None  # Reset timer if eyes are open

    else:
        closed_start = None

    # Check if eyes have been closed continuously for more than the sleep threshold
    if closed_start is not None:
        elapsed = time.time() - closed_start
        if elapsed >= SLEEP_THRESHOLD:
            asleep_text = "Yes"
        else:
            asleep_text = "No"

    # Create the response in the required format
    response = {
        "face_position": face_position,
        "eyes": eyes,
        "status": status,
        "asleep": asleep_text
    }

    return response

if __name__ == "__main__":
    