from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import os
import cv2

app = FastAPI()

# Allow frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://7ddd95.csb.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Healthcheck endpoint - responds instantly
@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

# Lazy model load (None until first request)
model = None

def remove_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

@app.post("/upload_video")
def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    global model
    if model is None:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("YOLO model loaded.")

    # Validate file extension
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid video file format")

    # Save uploaded file to temp path
    input_temp = NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".mp4")
    try:
        contents = file.file.read()
        input_temp.write(contents)
        input_temp.close()
        input_path = input_temp.name
    finally:
        file.file.close()

    # Open video file
    cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        os.remove(input_path)
        raise HTTPException(status_code=500, detail="Error opening video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if width <= 0 or height <= 0:
        cap.release()
        os.remove(input_path)
        raise HTTPException(status_code=500, detail="Invalid video dimensions")

    output_temp = NamedTemporaryFile(delete=False, suffix=".mp4")
    output_temp.close()
    output_path = output_temp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        os.remove(input_path)
        raise HTTPException(status_code=500, detail="Error initializing video writer")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(frame, persist=True)
            annotated = results[0].plot()
            out.write(annotated)
            frame_count += 1

        if frame_count == 0:
            raise RuntimeError("No frames processed")
    finally:
        cap.release()
        out.release()
        os.remove(input_path)

    background_tasks.add_task(remove_file, output_path)
    return FileResponse(output_path, media_type="video/mp4", filename="processed_video.mp4")
