# main.py
import os
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# -----------------------------
# Perf & safety knobs (set early)
# -----------------------------
os.environ.setdefault("ULTRALYTICS_NOAUTOINSTALL", "1")  # forbid runtime pip installs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("UVICORN_WORKERS", "1")  # keep single worker

# -----------------------------
# Optional S3 setup (do not fail import)
# -----------------------------
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = (os.getenv("S3_PREFIX") or "").strip("/")

try:
    import boto3  # type: ignore
    s3 = boto3.client("s3", region_name=AWS_REGION) if AWS_REGION else None
except Exception:
    boto3 = None  # type: ignore
    s3 = None

# -----------------------------
# FastAPI app & CORS
# -----------------------------
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

# -----------------------------
# Health / root (must be trivial)
# -----------------------------
@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.get("/")
def root():
    return {"status": "ok"}

# Explicit preflights (belt & suspenders)
@app.options("/presign-upload")
def options_presign():
    return PlainTextResponse("", status_code=200)

@app.options("/process_s3")
def options_process():
    return PlainTextResponse("", status_code=200)

# -----------------------------
# Pydantic models
# -----------------------------
class PresignResponse(BaseModel):
    uploadUrl: str
    key: str

class ProcessRequest(BaseModel):
    key: str = Field(..., description="S3 object key of the uploaded source video")
    imgsz: int = Field(320, ge=128, le=1280)
    vid_stride: int = Field(2, ge=1, le=10)
    conf: float = Field(0.30, ge=0.05, le=0.95)
    iou: float = Field(0.45, ge=0.1, le=0.9)
    classes: Optional[list[int]] = Field(None, description="Optional class ids filter")
    tracker: Literal["bytetrack.yaml"] = "bytetrack.yaml"  # avoids lapx auto-install

    @validator("key")
    def key_not_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("key must be a non-empty string")
        return v

class ProcessResponse(BaseModel):
    videoUrl: str
    processedKey: str

# -----------------------------
# Lazy YOLO (no heavy work at import)
# -----------------------------
_model = None
def get_model():
    """
    Lazy-load YOLO and clamp thread usage for small CPU instances.
    """
    global _model
    if _model is None:
        from ultralytics import YOLO  # heavy import only when needed
        _model = YOLO("yolov8n.pt")   # nano model = fastest on CPU
        try:
            _model.fuse()
        except Exception:
            pass
        # Clamp threads
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
        print("YOLO ready")
    return _model

# -----------------------------
# 1) Presign PUT URL for direct browser upload to S3
# -----------------------------
@app.get("/presign-upload", response_model=PresignResponse)
def presign_upload(ext: str = "mp4"):
    if ext.lower() not in {"mp4", "mov", "avi", "mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video extension")

    if not (S3_BUCKET and AWS_REGION and s3):
        raise HTTPException(
            status_code=500,
            detail="S3 is not configured (missing AWS_REGION/S3_BUCKET or boto3).",
        )

    key = f"{S3_PREFIX + '/' if S3_PREFIX else ''}uploads/{uuid.uuid4()}.{ext.lower()}"
    try:
        url = s3.generate_presigned_url(
            ClientMethod="putObject",
            Params={
                "Bucket": S3_BUCKET,
                "Key": key,
                # Safe default; browsers will still upload MOV/AVI fine
                "ContentType": "video/mp4",
            },
            ExpiresIn=15 * 60,  # 15 minutes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create presigned URL: {e}")

    return PresignResponse(uploadUrl=url, key=key)

# -----------------------------
# 2) Download from S3 → YOLO track → upload to S3 → return presigned GET
# -----------------------------
@app.post("/process_s3", response_model=ProcessResponse)
def process_s3(req: ProcessRequest):
    if not (S3_BUCKET and AWS_REGION and s3):
        raise HTTPException(
            status_code=500,
            detail="S3 is not configured (missing AWS_REGION/S3_BUCKET or boto3).",
        )

    with TemporaryDirectory() as workdir:
        work = Path(workdir)
        in_path = work / "input.mp4"
        out_base = work / "runs"

        # 1) Download source from S3
        try:
            s3.download_file(S3_BUCKET, req.key, str(in_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download from S3: {e}")

        # 2) Run YOLO tracking using built-in video pipeline
        model = get_model()
        try:
            model.track(
                source=str(in_path),
                save=True,                 # write annotated video
                stream=False,              # run to completion
                project=str(out_base),     # base dir
                name="out",                # subfolder
                exist_ok=True,
                device="cpu",
                tracker=req.tracker,       # ByteTrack (no lapx)
                imgsz=req.imgsz,
                vid_stride=req.vid_stride,
                conf=req.conf,
                iou=req.iou,
                classes=req.classes,
                verbose=False,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during tracking: {e}")

        # 3) Locate produced video
        out_root = out_base / "out"
        produced = None
        for ext in (".mp4", ".mov", ".avi", ".mkv"):
            found = list(out_root.rglob(f"*{ext}"))
            if found:
                produced = found[0]
                break
        if not produced:
            raise HTTPException(status_code=500, detail="Processed video not found")

        # 4) Upload processed file to S3
        processed_key = f"{S3_PREFIX + '/' if S3_PREFIX else ''}processed/{uuid.uuid4()}.mp4"
        try:
            s3.upload_file(str(produced), S3_BUCKET, processed_key, ExtraArgs={"ContentType": "video/mp4"})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload processed video failed: {e}")

    # 5) Return presigned GET for streaming
    try:
        get_url = s3.generate_presigned_url(
            ClientMethod="getObject",
            Params={"Bucket": S3_BUCKET, "Key": processed_key},
            ExpiresIn=60 * 60,  # 1 hour
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create presigned GET URL: {e}")

    return ProcessResponse(videoUrl=get_url, processedKey=processed_key)
