from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import tempfile
import torch
import os

app = FastAPI()

yolo_model = None  # Global, lazy-loaded

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    global yolo_model

    # Lazy load YOLO model
    if yolo_model is None:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")  # Use yolov8n for speed

    # Save uploaded video temporarily
    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_in.write(await file.read())
    temp_in.close()

    # Output temp file
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_out.close()

    # OpenCV Video Read
    cap = cv2.VideoCapture(temp_in.name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_out.name, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO
        results = yolo_model(frame)

        # Draw detections
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    # Return annotated video
    return StreamingResponse(open(temp_out.name, "rb"), media_type="video/mp4")

