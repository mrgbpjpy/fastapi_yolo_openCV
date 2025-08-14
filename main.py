# Import the os module to interact with the operating system
# Docs: https://docs.python.org/3/library/os.html
import os

# Import uuid to generate unique identifiers
# Docs: https://docs.python.org/3/library/uuid.html
import uuid

# Import Path from pathlib for path manipulation
# Docs: https://docs.python.org/3/library/pathlib.html
from pathlib import Path

# Import TemporaryDirectory for creating temporary directories
# Docs: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
from tempfile import TemporaryDirectory

# Import FastAPI to create the web application
# Docs: https://fastapi.tiangolo.com/
from fastapi import FastAPI, HTTPException

# Import JSONResponse for returning JSON responses
# Docs: https://fastapi.tiangolo.com/tutorial/response-model/#using-jsonresponse
from fastapi.responses import JSONResponse

# Import CORSMiddleware for Cross-Origin Resource Sharing
# Docs: https://fastapi.tiangolo.com/tutorial/cors/
from fastapi.middleware.cors import CORSMiddleware

# Import boto3 for AWS S3-compatible client (used with Cloudflare R2)
# Docs: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
import boto3

# Import Config from botocore to customize boto3 client behavior
# Docs: https://botocore.amazonaws.com/v1/documentation/api/latest/reference/config.html
from botocore.config import Config

# Import logging for detailed error tracking
# Docs: https://docs.python.org/3/library/logging.html
import logging

# Configure logging with INFO level
# Docs: https://docs.python.org/3/library/logging.html#logging.basicConfig
logging.basicConfig(level=logging.INFO)

# Create a logger instance for this module
# Docs: https://docs.python.org/3/library/logging.html#logging.getLogger
logger = logging.getLogger(__name__)

# ---------- perf/safety knobs ----------

# Set environment variable to disable Ultralytics autoinstall
# Docs: https://docs.python.org/3/library/os.html#os.environ
os.environ.setdefault("ULTRALYTICS_NOAUTOINSTALL", "1")

# Set environment variable to limit OpenMP threads to 1
# Docs: https://docs.python.org/3/library/os.html#os.environ
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Set environment variable to limit OpenBLAS threads to 1
# Docs: https://docs.python.org/3/library/os.html#os.environ
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Set environment variable to limit MKL threads to 1
# Docs: https://docs.python.org/3/library/os.html#os.environ
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Set environment variable to limit NumExpr threads to 1
# Docs: https://docs.python.org/3/library/os.html#os.environ
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------- R2 / S3-compatible config (Cloudflare R2) ----------

# Get R2 endpoint URL from environment, removing trailing slash
# Docs: https://docs.python.org/3/library/os.html#os.getenv
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "").rstrip("/")

# Get R2 Access Key ID from environment
# Docs: https://docs.python.org/3/library/os.html#os.getenv
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")

# Get R2 Secret Access Key from environment
# Docs: https://docs.python.org/3/library/os.html#os.getenv
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

# Get R2 or S3 bucket name from environment, prioritizing R2_BUCKET
# Docs: https://docs.python.org/3/library/os.html#os.getenv
S3_BUCKET = os.getenv("R2_BUCKET") or os.getenv("S3_BUCKET")

# Get S3 prefix from environment, removing leading/trailing slashes
# Docs: https://docs.python.org/3/library/os.html#os.getenv
S3_PREFIX = os.getenv("S3_PREFIX", "").strip("/")

# Define function to check if R2 environment variables are set
# Docs: https://docs.python.org/3/reference/compound_stmts.html#function
def have_r2_env() -> bool:
    env_vars = {
        "R2_ENDPOINT_URL": R2_ENDPOINT_URL,
        "R2_ACCESS_KEY_ID": R2_ACCESS_KEY_ID,
        "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
        "R2_BUCKET": S3_BUCKET
    }
    missing = [k for k, v in env_vars.items() if not v]
    if missing:
        logger.error(f"Missing R2 environment variables: {missing}")
    return bool(R2_ENDPOINT_URL and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and S3_BUCKET)

# Initialize global S3 client variable
# Docs: https://docs.python.org/3/reference/simple_stmts.html#assignment
_s3 = None

# Define function to lazily initialize the S3 (R2) client
# Docs: https://docs.python.org/3/reference/compound_stmts.html#function
def get_s3():
    """Lazy init the S3 (R2) client; raise a clean 500 if not configured."""
    # Access global _s3 variable
    # Docs: https://docs.python.org/3/reference/simple_stmts.html#global
    global _s3
    if _s3 is None:
        if not have_r2_env():
            logger.error("R2 environment not configured")
            raise HTTPException(
                status_code=500,
                detail="R2 is not configured. Missing one of: R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET",
            )
        try:
            _s3 = boto3.client(
                "s3",
                endpoint_url=R2_ENDPOINT_URL,
                aws_access_key_id=R2_ACCESS_KEY_ID,
                aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                # Use path-style addressing for R2 compatibility
                # Docs: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
                region_name="auto",
            )
            logger.info(f"S3 client initialized for {R2_ENDPOINT_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize S3 client: {e}")
    return _s3

# Create FastAPI application instance
# Docs: https://fastapi.tiangolo.com/tutorial/first-steps/
app = FastAPI(title="YOLOv8 Large Video Pipeline")

# Add CORS middleware to the app
# Docs: https://fastapi.tiangolo.com/tutorial/cors/
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://7ddd95.csb.app", "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define health check endpoint
# Docs: https://fastapi.tiangolo.com/tutorial/first-steps/#path-operation-decorator
@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "r2_configured": have_r2_env()})

# Define index endpoint to verify routes
# Docs: https://fastapi.tiangolo.com/tutorial/first-steps/#path-operation-decorator
@app.get("/")
def index():
    return {
        "ok": True,
        "routes": [getattr(r, "path", str(r)) for r in app.router.routes],
        "r2_configured": have_r2_env(),
        "bucket": S3_BUCKET,
        "prefix": S3_PREFIX,
    }

# ---------- YOLO (lazy) ----------

# Initialize global YOLO model variable
# Docs: https://docs.python.org/3/reference/simple_stmts.html#assignment
_model = None

# Define function to lazily load the YOLO model
# Docs: https://docs.python.org/3/reference/compound_stmts.html#function
def get_model():
    global _model
    if _model is None:
        # Import YOLO from ultralytics
        # Docs: https://docs.ultralytics.com/
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
        try:
            # Fuse the model for optimization
            # Docs: https://docs.ultralytics.com/modes/predict/#model-fusion
            _model.fuse()
        except Exception:
            pass
        try:
            # Import torch and set thread limits
            # Docs: https://pytorch.org/docs/stable/index.html
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        try:
            # Import cv2 and set thread limits
            # Docs: https://docs.opencv.org/4.x/
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

# 1) Presign a PUT for direct browser upload to R2
@app.get("/presign-upload")
def presign_upload(ext: str = "mp4", content_type: str = "video/mp4"):
    if not have_r2_env():
        logger.error("R2 environment check failed")
        raise HTTPException(500, "R2 not configured on server")
    ext = (ext or "mp4").lower()
    if ext not in {"mp4", "mov", "avi", "mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video extension")
    key = f"{S3_PREFIX + '/' if S3_PREFIX else ''}uploads/{uuid.uuid4()}.{ext}"
    s3 = get_s3()
    try:
        url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": S3_BUCKET,
                "Key": key,
                "ContentType": content_type,
            },
            ExpiresIn=900,
        )
        logger.info(f"Presigned URL generated for key: {key}")
    except Exception as e:
        logger.error(f"Presign upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to presign upload: {e}")
    return {"uploadUrl": url, "key": key, "bucket": S3_BUCKET}

# 2) Process uploaded object and return a presigned GET for the result
@app.post("/process_s3")
def process_s3(payload: dict):
    if not have_r2_env():
        logger.error("R2 environment check failed")
        raise HTTPException(500, "R2 not configured on server")
    key = payload.get("key")
    if not key or not isinstance(key, str):
        raise HTTPException(status_code=400, detail="Missing S3 key")
    imgsz = int(payload.get("imgsz", 320))
    vid_stride = int(payload.get("vid_stride", 2))
    s3 = get_s3()
    with TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        in_path = tmp / "input.mp4"
        out_base = tmp / "runs"
        try:
            s3.download_file(S3_BUCKET, key, str(in_path))
            logger.info(f"Downloaded file from R2: {key}")
        except Exception as e:
            logger.error(f"Download from R2 failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download from R2: {e}")
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
                tracker="bytetrack.yaml",
                imgsz=imgsz,
                vid_stride=vid_stride,
                conf=0.30,
                iou=0.45,
                verbose=False,
            )
            logger.info("YOLO tracking completed")
        except Exception as e:
            logger.error(f"Tracking error: {e}")
            raise HTTPException(status_code=500, detail=f"Tracking error: {e}")
        out_root = out_base / "out"
        produced = None
        for suf in (".mp4", ".mov", ".avi", ".mkv"):
            found = list(out_root.rglob(f"*{suf}"))
            if found:
                produced = found[0]
                break
        if not produced:
            logger.error("Processed video not found")
            raise HTTPException(status_code=500, detail="Processed video not found")
        processed_key = f"{S3_PREFIX + '/' if S3_PREFIX else ''}processed/{uuid.uuid4()}.mp4"
        try:
            s3.upload_file(str(produced), S3_BUCKET, processed_key, ExtraArgs={"ContentType": "video/mp4"})
            logger.info(f"Uploaded processed file to R2: {processed_key}")
        except Exception as e:
            logger.error(f"Upload to R2 failed: {e}")
            raise HTTPException(status_code=500, detail=f"Upload to R2 failed: {e}")
    try:
        get_url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": processed_key},
            ExpiresIn=3600,
        )
        logger.info(f"Presigned GET URL generated for: {processed_key}")
    except Exception as e:
        logger.error(f"Presign GET failed: {e}")
        raise HTTPException(status_code=500, detail=f"Presign GET failed: {e}")
    return {"videoUrl": get_url, "processedKey": processed_key}
