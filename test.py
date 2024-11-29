from ultralytics import YOLO
import cv2

# Load the YOLO model for detection
model = YOLO("best.pt")  # Ensure "best.pt" is a detection model
class_names = model.names

# Load the video
cap = cv2.VideoCapture('p.mp4')
count = 0

while True:
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    # Skip frames for faster processing
    if count % 3 != 0:
        continue

    # Resize the image for display purposes (optional)
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape

    # Run detection
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  # Boxes object containing the detection results

        # Loop through each detected box
        for box in boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]  # Top-left and bottom-right corners
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get the class ID and confidence score
            class_id = int(box.cls)
            conf = box.conf[0]  # Confidence score
            class_name = class_names[class_id]

            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Put the class name and confidence score on the bounding box
            cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the video with detections
    cv2.imshow('Pothole Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
