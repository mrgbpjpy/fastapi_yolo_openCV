from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import os

app = FastAPI(title="YOLOv8 Video Processor")

# --- CORS (note: no trailing slash) ---
ALLOWED_ORIGINS = ["https://7ddd95.csb.app", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health & root ---
@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.get("/")
def root():
    return {"status": "ok"}

# --- Explicit preflight for picky proxies ---
@app.options("/upload_video")
def options_upload_video():
    # CORSMiddleware will attach the Access-Control-* headers
    return PlainTextResponse("", status_code=200)

# --- Lazy model holder ---
_model = None
def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")  # small + fast
        print("YOLO loaded")
    return _model

def remove_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

# --- Main endpoint ---
@app.post("/upload_video")
def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    import cv2  # lazy import so startup is instant

    # Basic filename check
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video file format")

    # Save upload to temp
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    inp = NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        inp.write(file.file.read())
        inp.close()
        in_path = inp.name
    finally:
        file.file.close()

    # Open video
    cap = cv2.VideoCapture(in_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        os.remove(in_path)
        raise HTTPException(status_code=500, detail="Error opening video file")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if w <= 0 or h <= 0:
        cap.release(); os.remove(in_path)
        raise HTTPException(status_code=500, detail="Invalid video dimensions")

    # Output temp file
    out_tmp = NamedTemporaryFile(delete=False, suffix=".mp4")
    out_tmp.close()
    out_path = out_tmp.name

    # Writer (CPU-friendly)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release(); os.remove(in_path)
        raise HTTPException(status_code=500, detail="Error initializing video writer")

    model = get_model()
    frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            results = model.track(frame, persist=True)
            annotated = results[0].plot()
            out.write(annotated)
            frames += 1
        if frames == 0:
            raise RuntimeError("No frames processed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {e}")
    finally:
        cap.release(); out.release(); os.remove(in_path)

    background_tasks.add_task(remove_file, out_path)
    return FileResponse(out_path, media_type="video/mp4", filename="processed_video.mp4")
