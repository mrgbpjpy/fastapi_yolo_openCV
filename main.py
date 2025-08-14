# ---- hardening & perf before any heavy imports ----
import os
# Stop Ultralytics from pip-installing at runtime (e.g., lapx)
os.environ.setdefault("ULTRALYTICS_NOAUTOINSTALL", "1")
# Keep CPU thread usage sane on small Railway CPUs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("UVICORN_WORKERS", "1")  # belt & suspenders

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import TemporaryDirectory
from pathlib import Path
import shutil

app = FastAPI(title="YOLOv8 Video Processor")

# ---------- CORS ----------
ALLOWED_ORIGINS = ["https://7ddd95.csb.app", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- health & root ----------
@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.get("/")
def root():
    return {"status": "ok"}

# Explicit preflight (some edges are picky)
@app.options("/upload_video")
def options_upload_video():
    return PlainTextResponse("", status_code=200)

# ---------- lazy model + perf tuning ----------
_model = None
def get_model():
    global _model
    if _model is None:
        # lazy import to keep startup instant
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")  # nano model -> fastest
        try:
            _model.fuse()  # small CPU boost
        except Exception:
            pass

        # Also limit threads inside the process libs after import
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

        print("YOLO loaded (no auto-install).")
    return _model

# ---------- video endpoint (Ultralytics pipeline) ----------
@app.post("/upload_video")
def upload_video(file: UploadFile = File(...)):
    # Basic filename check
    name = (file.filename or "video.mp4")
    lower = name.lower()
    if not lower.endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(400, "Invalid video file format")

    # Work in a throwaway directory and then move the result to a stable temp path
    with TemporaryDirectory() as workdir:
        work = Path(workdir)
        in_dir = work / "input"
        out_dir = work / "runs"
        in_dir.mkdir(parents=True, exist_ok=True)

        in_path = in_dir / name
        try:
            in_bytes = file.file.read()
            in_path.write_bytes(in_bytes)
        except Exception as e:
            raise HTTPException(500, f"Error saving upload: {e}")
        finally:
            try:
                file.file.close()
            except Exception:
                pass

        model = get_model()

        # Use Ultralytics' internal video IO for speed; force ByteTrack to avoid lapx
        try:
            model.track(
                source=str(in_path),
                save=True,               # write annotated video to disk
                stream=False,            # run to completion
                project=str(out_dir),    # base output dir
                name="out",              # subfolder
                exist_ok=True,
                device="cpu",
                tracker="bytetrack.yaml",# <- avoids lapx (OCSORT) dependency
                imgsz=320,               # smaller -> faster CPU
                vid_stride=2,            # process every 2nd frame
                conf=0.30,
                iou=0.45,
                verbose=False,
            )
        except Exception as e:
            # If Ultralytics tries to auto-install, our env flag will prevent it and raise
            raise HTTPException(500, f"Error during tracking: {e}")

        # Locate output (Ultralytics mirrors input filename under runs/out/)
        out_root = out_dir / "out"
        candidates = []
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            candidates += list(out_root.rglob(ext))
        if not candidates:
            raise HTTPException(500, "Failed to locate processed video output")
        produced = candidates[0]

        # Move to a stable temp file (outside context manager) so FileResponse can stream it
        final_dir = Path(TemporaryDirectory().name)
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / "processed_video.mp4"
        shutil.move(str(produced), final_path)

    return FileResponse(
        str(final_path),
        media_type="video/mp4",
        filename="processed_video.mp4",
        headers={"Content-Disposition": "attachment; filename=processed_video.mp4"},
    )
