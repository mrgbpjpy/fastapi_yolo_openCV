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

# Import cv2 for OpenCV frame processing
# Docs: https://docs.opencv.org/4.x/
import cv2

# Import numpy for array operations
# Docs: https://numpy.org/doc/stable/
import numpy as np

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

# ---------- YOLO and OpenCV Processing ----------

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
        _model = YOLO("yolov8n.pt")  # Pre-trained on COCO dataset with 80 classes
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
            # Import cv2 for OpenCV frame processing
            # Docs: https://docs.opencv.org/4.x/
            import cv2
            cv2.setNumThreads(1)
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
        except Exception:
            pass
        logger.info("YOLO model loaded and ready")
    return _model

# Function to process each frame with YOLO and OpenCV
# Docs: https://docs.python.org/3/reference/compound_stmts.html#function
def process_frame(frame, results):
    # Convert frame to numpy array if not already
    # Docs: https://numpy.org/doc/stable/reference/arrays.html
    frame = np.array(frame) if not isinstance(frame, np.ndarray) else frame.copy()

    # Define all 80 COCO class names for labeling
    # Docs: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/coco128.yaml
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    # Draw bounding boxes and labels from YOLO tracking results
    # Docs: https://docs.ultralytics.com/modes/track/#track-results
    if results and results[0].boxes and results[0].boxes.xyxy is not None:
        for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            # Ensure class index is within valid range
            # Docs: https://numpy.org/doc/stable/reference/arrays.indexing.html
            cls_id = int(cls) if 0 <= int(cls) < len(coco_classes) else -1
            class_name = coco_classes[cls_id] if cls_id >= 0 else f"Unknown_{cls_id}"
            label = f"{class_name} ({conf:.2f})"
            # Draw rectangle for the bounding box
            # Docs: https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label text with background for readability
            # Docs: https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47c573f65d2a8c4d1daf0f26e2a
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 6, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

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
    imgsz = int(payload.get("imgsz", 640))  # Default to 640 for better detection
    vid_stride = int(payload.get("vid_stride", 1))  # Process every frame by default
    s3 = get_s3()
    with TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        in_path = tmp / "input.mp4"
        out_path = tmp / "output_annotated.mp4"
        try:
            s3.download_file(S3_BUCKET, key, str(in_path))
            logger.info(f"Downloaded file from R2: {key}")
        except Exception as e:
            logger.error(f"Download from R2 failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download from R2: {e}")
        model = get_model()
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            logger.error("Failed to open video file")
            raise HTTPException(status_code=500, detail="Failed to open video file")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 24.0  # Fallback FPS if metadata is missing
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % vid_stride == 0:  # Process every vid_stride frame
                results = model.track(source=frame, imgsz=imgsz, conf=0.25, iou=0.4, verbose=False)
                annotated_frame = process_frame(frame, results)
                out.write(annotated_frame)
                if results[0].boxes.xyxy.numel() > 0:
                    logger.info(f"Processed frame {frame_count} with {results[0].boxes.xyxy.shape[0]} detections")
            else:
                out.write(frame)  # Write unprocessed frame to maintain video length
            frame_count += 1
        cap.release()
        out.release()
        logger.info(f"Frame-by-frame processing with YOLO tracking and OpenCV completed for {frame_count} frames")
        produced = out_path
        if not produced.exists():
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
