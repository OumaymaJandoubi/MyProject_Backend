from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import uuid
import shutil

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load YOLO model
model = YOLO("best.pt")
class_names = model.names

# Create directories for storing processed images
PROCESSED_FOLDER = 'processed'

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Path to the log file where the location data is stored
LOG_FILE_PATH = "pothole_locations.txt"

@app.route('/detect', methods=['POST'])
def detect_potholes():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    if 'latitude' not in request.form or 'longitude' not in request.form:
        return jsonify({'error': 'Location data missing'}), 400

    # Get the image
    file = request.files['image']
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"

    # Save the uploaded image temporarily for processing
    input_path = f"temp_{unique_filename}"
    file.save(input_path)

    # Get the location data
    latitude = request.form['latitude']
    longitude = request.form['longitude']

    # Convert coordinates to an address using geopy
    geolocator = Nominatim(user_agent="pothole_detection_app")
    try:
        location = geolocator.reverse((latitude, longitude), language="en", timeout=10)
        address = location.address if location else "Address not found"
    except GeocoderTimedOut:
        address = "Address lookup timed out"

    # Save the location to the log file (with append mode to prevent modification)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(f"Detected at: {latitude}, {longitude} - Address: {address}\n")
    except Exception as e:
        return jsonify({'error': 'Error writing to location log file'}), 500

    # Process the image using YOLO
    img = cv2.imread(input_path)
    results = model.predict(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls)
            conf = box.conf[0]
            class_name = class_names[class_id]

            # Draw bounding boxes and labels on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 5)
            cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save the processed image to the PROCESSED_FOLDER
    processed_image_path = os.path.join(PROCESSED_FOLDER, f"processed_{unique_filename}")
    cv2.imwrite(processed_image_path, img)

    # Log processed image path for debugging (optional)
    print(f"Processed image saved at: {processed_image_path}")

    # Encode the processed image into bytes to send it back in the response
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    # Clean up the temporary uploaded file (optional)
    os.remove(input_path)



    # Return the processed image as a response
    return send_file(BytesIO(img_bytes), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
