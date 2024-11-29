from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
from ultralytics import YOLO

app = FastAPI()

# Load your YOLO model
model = YOLO("best.pt")

@app.post("/detect")
async def detect_potholes(video: UploadFile = File(...)):
    try:
        # Save the uploaded video
        video_path = f"temp_{video.filename}"
        with open(video_path, "wb") as f:
            f.write(await video.read())

        # Open the video with OpenCV
        cap = cv2.VideoCapture(video_path)
        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            results = model.predict(frame)
            for r in results:
                for box in r.boxes:
                    # Extract bounding box coordinates and class ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    conf = box.conf[0]
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class": model.names[class_id],
                        "confidence": float(conf)
                    })

        cap.release()
        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
