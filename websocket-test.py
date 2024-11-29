from flask import Flask
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2
import numpy as np
from hashlib import sha256

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow cross-origin requests

model = YOLO("best.pt")  # Load your trained model

cumulative_pothole_count = 0  # Persistent count across frames
processed_frames = set()  # Store hashes of processed frames


@socketio.on('frame')
def handle_frame(data):
    global cumulative_pothole_count

    try:
        # Convert the data to a numpy array and calculate its hash
        file_bytes = np.frombuffer(data, np.uint8)
        if file_bytes.size == 0:
            emit('response', {"error": "No data received"})
            return

        frame_hash = sha256(file_bytes).hexdigest()

        # Avoid processing the same frame again
        if frame_hash in processed_frames:
            emit('response', {"pothole_count": cumulative_pothole_count})
            return

        processed_frames.add(frame_hash)

        # Decode the image
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            emit('response', {"error": "Invalid frame"})
            return

        # Detect potholes in the frame
        results = model.predict(frame)
        current_frame_count = 0

        for r in results:
            for box in r.boxes:
                if int(box.cls) == 0:  # Assuming 'pothole' is class 0
                    current_frame_count += 1

        cumulative_pothole_count += current_frame_count

        # Send the updated count back to the frontend
        emit('response', {"pothole_count": cumulative_pothole_count})
    except Exception as e:
        emit('response', {"error": str(e)})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, debug=True)
