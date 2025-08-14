from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import os
import cv2
from ultralytics import YOLO

app = FastAPI()

# Healthcheck for Railway
@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://7ddd95.csb.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model once
model = YOLO("yolov8n.pt")

def remove_file(path: str):
    """Background task to clean up temporary files."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

@app.post("/upload_video")
def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Validate extension
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid video file format")

    # Save input temp file
    input_temp = NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".mp4")
    try:
        contents = file.file.read()
        input_temp.write(contents)
        input_temp.close()
        input_path = input_temp.name
    finally:
        file.file.close()

    # Open with FFmpeg backend
    cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        os.remove(input_path)
        raise HTTPException(status_code=500, detail="Error opening video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps} FPS, frames={total_frames}")

    if width <= 0 or height <= 0:
        cap.release()
        os.remove(input_path)
        raise HTTPException(status_code=500, detail="Invalid video dimensions")

    # Output file
    output_temp = NamedTemporaryFile(delete=False, suffix=".mp4")
    output_temp.close()
    output_path = output_temp.name

    # Use mp4v for broad CPU-only compatibility
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
        print(f"Processed {frame_count} frames")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {e}")
    finally:
        cap.release()
        out.release()
        os.remove(input_path)

    background_tasks.add_task(remove_file, output_path)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="processed_video.mp4",
        headers={"Content-Disposition": "attachment; filename=processed_video.mp4"}
    )
