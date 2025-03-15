from flask import Flask, Response
import cv2
import mediapipe as mp
import math

# Initialize Flask
app = Flask(__name__)

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

# Function to compute Euclidean distance between two normalized points
def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

# Route to stream video frames
def gen():
    while True:
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
                left_horizontal = euclidean_distance(face_landmarks.landmark[33], face_landmarks.landmark[133])
                left_vertical = euclidean_distance(face_landmarks.landmark[159], face_landmarks.landmark[145])
                left_EAR = left_vertical / left_horizontal

                right_horizontal = euclidean_distance(face_landmarks.landmark[362], face_landmarks.landmark[263])
                right_vertical = euclidean_distance(face_landmarks.landmark[386], face_landmarks.landmark[374])
                right_EAR = right_vertical / right_horizontal

                avg_EAR = (left_EAR + right_EAR) / 2.0

                EAR_threshold = 0.4  # Adjust threshold value if needed

                if avg_EAR < EAR_threshold:
                    status = "Eyes Closed"
                else:
                    status = "Eyes Open"

                # Display the status on the frame
                cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # Yield the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route to serve the video stream
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
