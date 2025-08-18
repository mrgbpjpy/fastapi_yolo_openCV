# -*- coding: utf-8 -*-
"""
FastAPI + R2 (S3-compatible) video pipeline:
- Presign PUT for direct upload from browser
- Download uploaded object
- Run YOLO per-frame overlays
- Produce web-safe MP4 (H.264 + AAC + faststart) via FFmpeg
- Upload processed object to R2 with correct metadata
- Return presigned GET (inline, video/mp4)
"""

import os
import uuid
import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import boto3
from botocore.config import Config

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------- logging & perf knobs ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

os.environ.setdefault("ULTRALYTICS_NOAUTOINSTALL", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------- R2 / S3 config ----------
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").rstrip("/")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("R2_BUCKET") or os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "").strip("/")
# optional: public site base like https://pub-XXXX.r2.dev
R2_PUBLIC_BASE = os.getenv("R2_PUBLIC_BASE", "").rstrip("/")

def have_r2_env() -> bool:
    ok = bool(R2_ENDPOINT_URL and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and S3_BUCKET)
    if not ok:
        missing = [k for k, v in {
            "R2_ENDPOINT_URL": R2_ENDPOINT_URL,
            "R2_ACCESS_KEY_ID": R2_ACCESS_KEY_ID,
            "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
            "R2_BUCKET": S3_BUCKET
        }.items() if not v]
        logger.error("Missing R2 env: %s", missing)
    return ok

_s3 = None
def get_s3():
    global _s3
    if _s3 is None:
        if not have_r2_env():
            raise HTTPException(500, "R2 not configured")
        try:
            _s3 = boto3.client(
                "s3",
                endpoint_url=R2_ENDPOINT_URL,
                aws_access_key_id=R2_ACCESS_KEY_ID,
                aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
                region_name="auto",
            )
            logger.info("S3 client initialized for %s", R2_ENDPOINT_URL)
        except Exception as e:
            logger.exception("Failed to init S3")
            raise HTTPException(500, f"Failed to initialize S3 client: {e}")
    return _s3

# ---------- FastAPI ----------
app = FastAPI(title="YOLOv8 Large Video Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://84fl4c.csb.app",
        "https://7ddd95.csb.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "r2_configured": have_r2_env(), "bucket": S3_BUCKET})

@app.get("/")
def index():
    return {
        "ok": True,
        "routes": [getattr(r, "path", str(r)) for r in app.router.routes],
        "r2_configured": have_r2_env(),
        "bucket": S3_BUCKET,
        "prefix": S3_PREFIX,
        "public_base": R2_PUBLIC_BASE or None,
    }

# ---------- YOLO ----------
_model = None
def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
        try:
            _model.fuse()
        except Exception:
            pass
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        try:
            cv2.setNumThreads(1)
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
        logger.info("YOLO model loaded")
    return _model

COCO = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def draw_boxes(frame: np.ndarray, results) -> np.ndarray:
    f = frame.copy()
    if results and results[0].boxes and results[0].boxes.xyxy is not None:
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            idx = int(cls)
            name = COCO[idx] if 0 <= idx < len(COCO) else f"cls_{idx}"
            label = f"{name} ({conf:.2f})"
            cv2.rectangle(f, (x1, y1), (x2, y2), (0,255,0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(f, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0,0,0), -1)
            cv2.putText(f, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return f

# ---------- FFmpeg helpers ----------
def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False

def ffmpeg_h264_faststart(inp: Path, out: Path) -> Tuple[bool, Optional[str]]:
    """
    Transcode/remux to a browser-safe MP4:
    - H.264 video (YUV420), AAC audio
    - moov atom moved to front (-movflags +faststart)
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(inp),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.0",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k", "-ac", "2",
        str(out)
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        ok = (r.returncode == 0) and out.exists() and out.stat().st_size > 0
        return ok, (r.stderr or r.stdout)
    except Exception as e:
        return False, str(e)

# ---------- API: presign PUT ----------
@app.get("/presign-upload")
def presign_upload(ext: str = "mp4", content_type: str = "video/mp4"):
    if not have_r2_env():
        raise HTTPException(500, "R2 not configured on server")
    ext = (ext or "mp4").lower()
    if ext not in {"mp4", "mov", "avi", "mkv"}:
        raise HTTPException(400, "Unsupported video extension")
    key = f"{S3_PREFIX + '/' if S3_PREFIX else ''}uploads/{uuid.uuid4()}.{ext}"
    s3 = get_s3()
    try:
        url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": S3_BUCKET, "Key": key, "ContentType": content_type},
            ExpiresIn=900,
        )
        logger.info("Presigned PUT for %s", key)
        return {"uploadUrl": url, "key": key, "bucket": S3_BUCKET}
    except Exception as e:
        logger.exception("Presign PUT failed")
        raise HTTPException(500, f"Failed to presign upload: {e}")

# ---------- API: process ----------
@app.post("/process_s3")
def process_s3(payload: dict):
    if not have_r2_env():
        raise HTTPException(500, "R2 not configured on server")
    key = payload.get("key")
    if not key or not isinstance(key, str):
        raise HTTPException(400, "Missing S3 key")
    imgsz = int(payload.get("imgsz", 640))
    vid_stride = int(payload.get("vid_stride", 1))

    s3 = get_s3()

    with TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        src_path = tmp / "input.mp4"
        ocv_path = tmp / "ocv_out.mp4"         # first pass (OpenCV writer)
        web_path = tmp / "web_out.mp4"         # final web-safe file

        # 1) download uploaded object
        try:
            s3.download_file(S3_BUCKET, key, str(src_path))
            logger.info("Downloaded from R2: %s", key)
        except Exception as e:
            logger.exception("Download failed")
            raise HTTPException(500, f"Failed to download from R2: {e}")

        # 2) run YOLO overlay, write via OpenCV
        model = get_model()
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise HTTPException(500, "Failed to open video file")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        # NOTE: OpenCV 'mp4v' often yields MPEG‑4 Part 2. We'll post-process with ffmpeg.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(ocv_path), fourcc, float(fps), (width, height))
        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_count % max(1, vid_stride) == 0:
                results = model.track(source=frame, imgsz=imgsz, conf=0.25, iou=0.4, verbose=False)
                frame = draw_boxes(frame, results)
            out.write(frame)
            frame_count += 1
        cap.release()
        out.release()
        logger.info("Frame processing done: %d frames", frame_count)

        # 3) make web‑safe via FFmpeg (H.264/AAC + faststart)
        final_path = ocv_path
        if have_ffmpeg():
            ok, log = ffmpeg_h264_faststart(ocv_path, web_path)
            if ok:
                final_path = web_path
                logger.info("FFmpeg produced web-safe MP4 (%s)", web_path.name)
            else:
                logger.warning("FFmpeg failed; using OpenCV file. log: %s", (log or "")[:4000])
        else:
            logger.warning("FFmpeg not available; using OpenCV file (may not play in Chrome).")

        # 4) upload processed object with correct metadata
        processed_key = f"{S3_PREFIX + '/' if S3_PREFIX else ''}processed/{uuid.uuid4()}.mp4"
        try:
            s3.upload_file(
                str(final_path),
                S3_BUCKET,
                processed_key,
                ExtraArgs={
                    "ContentType": "video/mp4",
                    "CacheControl": "public, max-age=3600",
                },
            )
            logger.info("Uploaded processed file: %s", processed_key)
        except Exception as e:
            logger.exception("Upload to R2 failed")
            raise HTTPException(500, f"Upload to R2 failed: {e}")

    # 5) return playback URL
    try:
        if R2_PUBLIC_BASE:
            # If you’ve set your public site base (https://pub-XXXX.r2.dev),
            # you can return a stable public URL.
            public_url = f"{R2_PUBLIC_BASE}/{processed_key}"
            return {"videoUrl": public_url, "processedKey": processed_key, "public": True}

        # Otherwise, return presigned GET with inline/video headers
        get_url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": S3_BUCKET,
                "Key": processed_key,
                "ResponseContentType": "video/mp4",
                "ResponseContentDisposition": "inline",
            },
            ExpiresIn=3600,
        )
        logger.info("Presigned GET for: %s", processed_key)
        return {"videoUrl": get_url, "processedKey": processed_key, "public": False}
    except Exception as e:
        logger.exception("Presign GET failed")
        raise HTTPException(500, f"Presign GET failed: {e}")
