from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import os

app = FastAPI()

# CORS (no trailing slash)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://7ddd95.csb.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health(): return JSONResponse({"status":"ok"})

@app.get("/")
def root(): return {"status":"ok"}

# Preflight (belt & suspenders)
@app.options("/upload_video")
def options_upload_video():
    return PlainTextResponse("", 200)

_model = None
def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
        print(">>> YOLO loaded")
    return _model

def remove_file(p):
    try: os.remove(p)
    except FileNotFoundError: pass

@app.post("/upload_video")
def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    import cv2  # lazy import keeps startup instant

    if not file.filename.lower().endswith((".mp4",".avi",".mov")):
        raise HTTPException(400, "Invalid video file format")

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    inp = NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        inp.write(file.file.read()); inp.close()
        in_path = inp.name
    finally:
        file.file.close()

    cap = cv2.VideoCapture(in_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        os.remove(in_path); raise HTTPException(500, "Error opening video file")

    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if w <= 0 or h <= 0:
        cap.release(); os.remove(in_path); raise HTTPException(500, "Invalid video dimensions")

    out_tmp = NamedTemporaryFile(delete=False, suffix=".mp4"); out_tmp.close()
    out_path = out_tmp.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release(); os.remove(in_path); raise HTTPException(500, "Error initializing video writer")

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
        if frames == 0: raise RuntimeError("No frames processed")
    except Exception as e:
        raise HTTPException(500, f"Error processing video: {e}")
    finally:
        cap.release(); out.release(); os.remove(in_path)

    background_tasks.add_task(remove_file, out_path)
    return FileResponse(out_path, media_type="video/mp4", filename="processed_video.mp4")
