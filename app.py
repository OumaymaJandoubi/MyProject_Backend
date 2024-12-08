import os
import uuid
import json
import cv2
import mysql.connector
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from web3 import Web3
from ultralytics import YOLO
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load YOLO model
try:
    model = YOLO("best.pt")  # Ensure 'best.pt' is in the correct path
    class_names = model.names
except Exception as e:
    raise RuntimeError(f"Error loading YOLO model: {e}")

# Create directories for storing processed images
PROCESSED_FOLDER = 'processed'
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Blockchain setup
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))  # Replace with your provider URL
if not w3.is_connected():
    raise ConnectionError("Failed to connect to Ethereum provider.")

# Load the deployed contract ABI and address
try:
    with open('abi.json', 'r') as f:  # Adjust path to your contract build file
        contract_data = json.load(f)
        contract_abi = contract_data.get('abi', None)
        if not isinstance(contract_abi, list):
            raise ValueError("ABI is not in the correct format.")
except FileNotFoundError:
    raise FileNotFoundError("abi.json file not found.")
except json.JSONDecodeError:
    raise ValueError("Error decoding abi.json. Ensure it's valid JSON.")

contract_address = "0x888C158C1bcbBB72F3499e380289f1DccD414244"
contract = w3.eth.contract(address=w3.to_checksum_address(contract_address), abi=contract_abi)

# Define account and private key for transactions
account = "0x2Fa990C431553b352F8A2b6783F2107302017F6c"
private_key = "0x242940ee87fa0e6cbce558adab48e7e02334842dc2a098b66f4d217a825b36f2"

# MySQL setup
db_connection = mysql.connector.connect(
    host="127.0.0.1",  # Your MySQL host
    user="root",  # Your MySQL username
    password="password",  # Your MySQL password
    database="pothole_detection"
)
cursor = db_connection.cursor()

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

    # Store the real address in a variable and print it to the console
    real_address = address
    print(f"Address of the location: {real_address}")

    try:
        # Use w3.to_wei to convert the gas price
        gas_price = w3.to_wei('50', 'gwei')
        tx = contract.functions.storeLocation(real_address).transact({
            'from': account,
            'gas': 2000000,
            'gasPrice': gas_price
        })

        # Wait for transaction receipt
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx)
        print(f"Transaction successful with hash: {tx_receipt.transactionHash.hex()}")

    except Exception as e:
        print(f"Error storing location in blockchain: {e}")
        return jsonify({'error': 'Failed to store location on blockchain'}), 500

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

    # Insert only address and image path into the MySQL database
    cursor.execute("INSERT INTO potholes (address, image_path) VALUES (%s, %s)",
                   (real_address, processed_image_path))
    db_connection.commit()

    # Encode the processed image into bytes to send it back in the response
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    # Clean up the temporary uploaded file
    os.remove(input_path)

    # Return the processed image as a response
    return send_file(BytesIO(img_bytes), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
