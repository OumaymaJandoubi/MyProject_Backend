from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from hashlib import sha256

app = Flask(__name__)
model = YOLO("best.pt")  # Load your trained model

cumulative_pothole_count = 0  # Persistent count across frames
processed_frames = set()  # Store hashes of processed frames


@app.route('/detect', methods=['POST'])
def detect():
    global cumulative_pothole_count

    # Get the image data and calculate its hash
    file_bytes = np.frombuffer(request.data, np.uint8)
    if file_bytes.size == 0:
        return jsonify({"error": "No data received"}), 400
    frame_hash = sha256(request.data).hexdigest()

    # Avoid processing the same frame again
    if frame_hash in processed_frames:
        return jsonify({"pothole_count": cumulative_pothole_count})

    # Add the frame hash to the set of processed frames
    processed_frames.add(frame_hash)

    # Decode the image
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Detect potholes in the frame
    results = model.predict(frame)
    current_frame_count = 0

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # Assuming 'pothole' is class 0
                current_frame_count += 1

    # Update the cumulative count
    cumulative_pothole_count += current_frame_count

    return jsonify({"pothole_count": cumulative_pothole_count})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
