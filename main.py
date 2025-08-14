import os
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Performance and safety configurations
os.environ.setdefault("ULTRALYTICS_NOAUTOINSTALL", "1")  # Prevent pip installs at runtime
os.environ.setdefault("OMP_NUM_THREADS", "1")  # Limit threads for CPU efficiency
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# FastAPI app setup with CORS
ALLOWED_ORIGINS = [
    "https://7ddd95.csb.app",
    "http://localhost:3000",
]
app = FastAPI(title="YOLOv8 Large Video Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health and root endpoints for Railway healthcheck
@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.get("/")
def root():
    return {"status": "ok"}

# Explicit preflight responses for CORS compatibility
@app.options("/upload-and-process")
def _opt1():
    return PlainTextResponse("", status_code=200)

# Schemas for request and response
class ProcessRequest(BaseModel):
    imgsz: int = Field(320, ge=128, le=1280, description="Image size for YOLO processing")
    vid_stride: int = Field(2, ge=1, le=10, description="Video frame stride")
    conf: float = Field(0.30, ge=0.05, le=0.95, description="Confidence threshold")
    iou: float = Field(0.45, ge=0.1, le=0.9, description="IoU threshold")
    classes: Optional[list[int]] = Field(None, description="Optional class IDs filter")
    tracker: Literal["bytetrack.yaml"] = "bytetrack.yaml"  # Use ByteTrack

class ProcessResponse(BaseModel):
    videoPath: str  # Path to processed video

# Lazy-load YOLO model to avoid heavy imports at startup
_model = None
def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")  # Use nano model for CPU speed
        try:
            _model.fuse()
        except Exception:
            pass
        # Clamp threads for responsiveness
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        try:
            import cv2
            cv2.setNumThreads(1)
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
        except Exception:
            pass
        print("[boot] YOLO ready")
    return _model

# Endpoint to upload and process video directly
@app.post("/upload-and-process", response_model=ProcessResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    imgsz: int = 320,
    vid_stride: int = 2,
    conf: float = 0.30,
    iou: float = 0.45,
    classes: Optional[list[int]] = None,
    tracker: str = "bytetrack.yaml"
):
    # Validate file extension
    ext = (file.filename.split(".")[-1] if file.filename else "mp4").lower()
    if ext not in {"mp4", "mov", "avi", "mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video extension")

    # Create temporary directory for processing
    with TemporaryDirectory() as workdir:
        work = Path(workdir)
        in_path = work / f"input.{ext}"
        out_base = work / "runs"

        # Save uploaded file
        try:
            with in_path.open("wb") as f:
                content = await file.read()
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

        # Run YOLO tracking
        model = get_model()
        try:
            model.track(
                source=str(in_path),
                save=True,  # Save annotated video
                stream=False,  # Run to completion
                project=str(out_base),  # Output directory
                name="out",  # Subfolder
                exist_ok=True,
                device="cpu",
                tracker=tracker,
                imgsz=imgsz,
                vid_stride=vid_stride,
                conf=conf,
                iou=iou,
                classes=classes,
                verbose=False,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during tracking: {e}")

        # Locate processed video
        out_root = out_base / "out"
        produced = None
        for ext in (".mp4", ".mov", ".avi", ".mkv"):
            found = list(out_root.rglob(f"*{ext}"))
            if found:
                produced = found[0]
                break
        if not produced:
            raise HTTPException(status_code=500, detail="Processed video not found")

        # Move processed video to a persistent location (Railway ephemeral storage workaround)
        processed_path = Path(f"/app/processed/{uuid.uuid4()}.mp4")
        processed_path.parent.mkdir(exist_ok=True)
        processed_path.write(produced.read_bytes())

    return ProcessResponse(videoPath=str(processed_path))

# Serve processed video
@app.get("/processed/{video_id}")
async def serve_processed(video_id: str):
    video_path = Path(f"/app/processed/{video_id}.mp4")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")
