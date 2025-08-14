from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from tempfile import NamedTemporaryFile
import os

app = FastAPI(title="YOLOv8 Video Processor")

# ---------------------------
# CORS (no trailing slashes)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://7ddd95.csb.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Optional: size guard (friendly 413 before work starts)
# Adjust MAX_MB as desired. Railway/edge proxies may still enforce their own cap.
# ---------------------------
class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        size = int(request.headers.get("content-length", "0") or 0)
        if size and size > self.max_bytes:
            mb = self.max_bytes // (1024 * 1024)
            raise HTTPException(status_code=413, detail=f"File too large (> {mb} MB)")
        return await call_next(request)

# Set to 25 MB by default to mirror common edge limits. Increase if your ingress allows it.
app.add_middleware(LimitUploadSize, max_bytes=25 * 1024 * 1024)

# ---------------------------
# Health & root
# ---------------------------
@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.get("/")
def root():
    return {"status": "ok"}

# ---------------------------
# Explicit preflight (belt & suspenders)
# ---------------------------
@app.options("/upload_video")
def options_upload_video():
    return PlainTextResponse("", status_code=200)

# ---------------------------
# Lazy model & helpers
# ---------------------------
_model = None

def get_model():
    """Lazy-load Ultralytics YOLO model on first request."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")  # nano model for speed
        print("YOLO model loaded.")
    return _model

def remove_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

# ---------------------------
# Main endpoint
# ---------------------------
@app.post("/upload_video")
def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accept a video file, run YOLOv8 tracking frame-by-frame,
    write an annotated MP4, and return it.
    """
    # Lazy import OpenCV so startup stays fast
    import cv2

    # Validate extension (quick sanity check)
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video file format")

    # Save uploaded input to a temp path
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    in_tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        in_bytes = file.file.read()
        in_tmp.write(in_bytes)
        in_tmp.close()
        in_path = in_tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving upload: {e}")
    finally:
        file.file.close()

    # Open with FFmpeg backend for better codec support
    cap = cv2.VideoCapture(in_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        os.remove(in_path)
        raise HTTPException(status_code=500, detail="Error opening video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if width <= 0 or height <= 0:
        cap.release()
        os.remove(in_path)
        raise HTTPException(status_code=500, detail="Invalid video dimensions")

    # Prepare output temp file
    out_tmp = NamedTemporaryFile(delete=False, suffix=".mp4")
    out_tmp.close()
    out_path = out_tmp.name

    # Use mp4v (widely available in CPU-only images)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        os.remove(in_path)
        raise HTTPException(status_code=500, detail="Error initializing video writer")

    model = get_model()
    frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO tracking (persist IDs across frames)
            results = model.track(frame, persist=True)
            annotated = results[0].plot()  # draw boxes, ids, etc.
            out.write(annotated)

            frames += 1

        if frames == 0:
            raise RuntimeError("No frames processed (unsupported codec or empty video)")
        print(f"Processed {frames} frames -> {out_path}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {e}")
    finally:
        cap.release()
        out.release()
        os.remove(in_path)

    # Clean up output after the response is sent
    background_tasks.add_task(remove_file, out_path)

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename="processed_video.mp4",
        headers={"Content-Disposition": "attachment; filename=processed_video.mp4"},
    )
