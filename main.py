# main.py
import os, uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import boto3
from botocore.config import Config

# --- perf/safety knobs ---
os.environ.setdefault("ULTRALYTICS_NOAUTOINSTALL", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---- R2 / S3-compatible config ----
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")  # e.g. https://<account>.r2.cloudflarestorage.com
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")      # not strictly needed if endpoint URL is set
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("R2_BUCKET") or os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "").strip("/")

if not (R2_ENDPOINT_URL and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and S3_BUCKET):
    raise RuntimeError("Missing R2/S3 environment vars. Required: R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.")

def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL.rstrip("/"),
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",  # R2 uses 'auto'
    )

s3 = make_s3()

app = FastAPI(title="YOLOv8 Large Video Pipeline")

# CORS (add your frontend origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://7ddd95.csb.app", "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

# ---- YOLO (lazy) ----
_model = None
def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
        try: _model.fuse()
        except: pass
        try:
            import torch
            torch.set_num_threads(1); torch.set_num_interop_threads(1)
        except: pass
        try:
            import cv2
            cv2.setNumThreads(1)
            try: cv2.ocl.setUseOpenCL(False)
            except: pass
        except: pass
        print("YOLO ready")
    return _model

# 1) Presign a PUT for direct browser upload to R2
@app.get("/presign-upload")
def presign_upload(ext: str = "mp4"):
    ext = (ext or "mp4").lower()
    if ext not in {"mp4", "mov", "avi", "mkv"}:
        raise HTTPException(400, "Unsupported video extension")

    key = f"{S3_PREFIX + '/' if S3_PREFIX else ''}uploads/{uuid.uuid4()}.{ext}"
    try:
        url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": S3_BUCKET,
                "Key": key,
                "ContentType": "video/mp4",  # safe default
            },
            ExpiresIn=900,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to presign upload: {e}")

    return {"uploadUrl": url, "key": key}

# 2) Process uploaded object and return a presigned GET for the result
@app.post("/process_s3")
def process_s3(payload: dict):
    key = payload.get("key")
    if not key or not isinstance(key, str):
        raise HTTPException(400, "Missing S3 key")

    imgsz = int(payload.get("imgsz", 320))
    vid_stride = int(payload.get("vid_stride", 2))

    from ultralytics.utils import LOGGER

    with TemporaryDirectory() as workdir:
        work = Path(workdir)
        in_path = work / "input.mp4"
        out_base = work / "runs"

        # Download from R2
        try:
            s3.download_file(S3_BUCKET, key, str(in_path))
        except Exception as e:
            raise HTTPException(500, f"Failed to download from R2: {e}")

        # Run YOLO tracking
        model = get_model()
        try:
            model.track(
                source=str(in_path),
                save=True,
                stream=False,
                project=str(out_base),
                name="out",
                exist_ok=True,
                device="cpu",
                tracker="bytetrack.yaml",  # avoids lapx GPU deps
                imgsz=imgsz,
                vid_stride=vid_stride,
                conf=0.30,
                iou=0.45,
                verbose=False,
            )
        except Exception as e:
            raise HTTPException(500, f"Tracking error: {e}")

        # Find output video
        out_root = out_base / "out"
        produced = None
        for ext in (".mp4", ".mov", ".avi", ".mkv"):
            found = list(out_root.rglob(f"*{ext}"))
            if found:
                produced = found[0]
                break
        if not produced:
            raise HTTPException(500, "Processed video not found")

        # Upload back to R2
        processed_key = f"{S3_PREFIX + '/' if S3_PREFIX else ''}processed/{uuid.uuid4()}.mp4"
        try:
            s3.upload_file(str(produced), S3_BUCKET, processed_key, ExtraArgs={"ContentType": "video/mp4"})
        except Exception as e:
            raise HTTPException(500, f"Upload to R2 failed: {e}")

    # Presign a GET to stream in the browser
    try:
        get_url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": processed_key},
            ExpiresIn=3600,
        )
    except Exception as e:
        raise HTTPException(500, f"Presign GET failed: {e}")

    return {"videoUrl": get_url, "processedKey": processed_key}
