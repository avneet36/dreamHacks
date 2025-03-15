import cv2
import mediapipe as mp
import math

# Function to compute Euclidean distance between two normalized points
def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# For drawing the landmarks (optional)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for a mirror-view and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Optionally draw the facial landmarks for visualization
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # --- Calculate Eye Aspect Ratio (EAR) for each eye ---
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

            # --- Determine Eye State ---
            # Set a threshold; if avg_EAR is less than this, we assume the eyes are closed.
            EAR_threshold = 0.2  # You might need to adjust this value

            if avg_EAR < EAR_threshold:
                status = "Eyes Closed"
            else:
                status = "Eyes Open"

            # Display the status on the frame
            cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
    else:
        # If no face is detected, print "Can't detect"
        cv2.putText(frame, "Can't detect", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    # Show the video feed with the text overlay
    cv2.imshow("Eye State Detection", frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup: Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
