from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import cv2
import numpy as np
import time
import eyesClose  # Import the eyesClose.py module

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your Flask app

# Initialize global variables
eyesClose.closed_start = None  # To track if the eyes are closed

@app.route('/process_video', methods=['POST'])
def process_video():
    # Receive video frame from the client
    file = request.files['file']  # Assuming the client sends the video frame as a file
    in_memory_file = file.read()  # Read the file into memory
    
    # Convert the byte stream into a numpy array (this is where OpenCV processes the image)
    np_arr = np.frombuffer(in_memory_file, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Pass the frame to the eyesClose.py code for processing
    response = eyesClose.detect_eye_sleep_status(frame)

    # Return the JSON response to the client
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
