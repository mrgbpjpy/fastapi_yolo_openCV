# main.py
import os, uuid, shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import boto3

# ---- perf & safety knobs for big jobs ----
os.environ.setdefault("ULTRALYTICS_NOAUTOINSTALL", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "").strip("/")
s3 = boto3.client("s3", region_name=AWS_REGION)

app = FastAPI(title="YOLOv8 Large Video Pipeline")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://7ddd95.csb.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

# 1) Get a presigned PUT for direct browser upload to S3
@app.get("/presign-upload")
def presign_upload(ext: str = "mp4"):
    if ext.lower() not in {"mp4", "mov", "avi", "mkv"}:
        raise HTTPException(400, "Unsupported video extension")

    key = f"{S3_PREFIX + '/' if S3_PREFIX else ''}uploads/{uuid.uuid4()}.{ext.lower()}"
    url = s3.generate_presigned_url(
        ClientMethod="putObject",
        Params={
            "Bucket": S3_BUCKET,
            "Key": key,
            "ContentType": f"video/{'mp4' if ext=='mp4' else 'mp4'}",  # browser accepts mp4
        },
        ExpiresIn=900,  # 15 min
    )
    return {"uploadUrl": url, "key": key}

# Lazy model (CPU-friendly)
_model = None
def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
        try: _model.fuse()
        except: pass
        # limit threads
        try:
            import torch; torch.set_num_threads(1); torch.set_num_interop_threads(1)
        except: pass
        try:
            import cv2; cv2.setNumThreads(1); 
            try: cv2.ocl.setUseOpenCL(False)
            except: pass
        except: pass
        print("YOLO ready")
    return _model

# 2) Process an uploaded S3 object and return a presigned GET for the result
@app.post("/process_s3")
def process_s3(payload: dict):
    """
    payload = { "key": "<s3 object key>", "imgsz": 320, "vid_stride": 2 }
    """
    key = payload.get("key")
    if not key or not isinstance(key, str):
        raise HTTPException(400, "Missing S3 key")

    imgsz = int(payload.get("imgsz", 320))
    vid_stride = int(payload.get("vid_stride", 2))

    # work dir
    with TemporaryDirectory() as workdir:
        work = Path(workdir)
        in_path = work / "input.mp4"
        out_base = work / "runs"

        # Download from S3
        try:
            s3.download_file(S3_BUCKET, key, str(in_path))
        except Exception as e:
            raise HTTPException(500, f"Failed to download from S3: {e}")

        # Run tracking using Ultralytics’ pipeline
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
                tracker="bytetrack.yaml",  # avoids lapx
                imgsz=imgsz,
                vid_stride=vid_stride,
                conf=0.30,
                iou=0.45,
                verbose=False,
            )
        except Exception as e:
            raise HTTPException(500, f"Error during tracking: {e}")

        # Locate result
        out_root = out_base / "out"
        produced = None
        for ext in (".mp4", ".mov", ".avi", ".mkv"):
            found = list(out_root.rglob(f"*{ext}"))
            if found:
                produced = found[0]; break
        if not produced:
            raise HTTPException(500, "Processed video not found")

        # Upload processed to S3 (e.g., under processed/)
        processed_key = f"{S3_PREFIX + '/' if S3_PREFIX else ''}processed/{uuid.uuid4()}.mp4"
        try:
            s3.upload_file(str(produced), S3_BUCKET, processed_key, ExtraArgs={"ContentType": "video/mp4"})
        except Exception as e:
            raise HTTPException(500, f"Upload processed video failed: {e}")

    # 3) Return a short-lived GET URL for streaming in the browser
    get_url = s3.generate_presigned_url(
        ClientMethod="getObject",
        Params={"Bucket": S3_BUCKET, "Key": processed_key},
        ExpiresIn=3600,  # 1 hour
    )
    return {"videoUrl": get_url, "processedKey": processed_key}
