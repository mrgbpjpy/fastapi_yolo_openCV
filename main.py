from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from tempfile import TemporaryDirectory
from pathlib import Path
import os
import shutil

app = FastAPI(title="YOLOv8 Video Processor")

# --- CORS (no trailing slash) ---
ALLOWED_ORIGINS = ["https://7ddd95.csb.app", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# --- Lazy model ---
_model = None
def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")  # nano for speed
        # Optional: fuse for a tiny CPU boost
        try:
            _model.fuse()
        except Exception:
            pass
        print("YOLO loaded")
    return _model

# --- Main endpoint: use Ultralytics pipeline to read/track/write ---
@app.post("/upload_video")
def upload_video(file: UploadFile = File(...)):
    # Very light validation
    name = (file.filename or "video.mp4").lower()
    if not name.endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(400, "Invalid video file format")

    # Create a throwaway working dir per request
    with TemporaryDirectory() as workdir:
        work = Path(workdir)
        in_path = work / "input" / name
        out_dir = work / "runs"   # Ultralytics will write under project/name/*
        in_path.parent.mkdir(parents=True, exist_ok=True)

        # Save upload
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

        # Run tracking internally (ByteTrack = no lapx dependency)
        model = get_model()
        try:
            model.track(
                source=str(in_path),
                stream=False,            # let it run and write outputs
                save=True,               # write annotated video
                project=str(out_dir),    # where to write
                name="out",              # subfolder
                exist_ok=True,
                imgsz=320,               # reduce pixels -> faster CPU
                vid_stride=2,            # process every 2nd frame
                conf=0.25,               # tweak as needed
                iou=0.45,
                device="cpu",
                tracker="bytetrack.yaml" # avoid OCSORT -> no lapx at runtime
            )
        except Exception as e:
            # If something goes wrong inside Ultralytics
            raise HTTPException(500, f"Error during tracking: {e}")

        # Find the produced video (Ultralytics mirrors input name)
        out_root = out_dir / "out"
        # pick the first mp4/mov/avi in output
        candidates = list(out_root.rglob("*.mp4")) + list(out_root.rglob("*.mov")) + list(out_root.rglob("*.avi"))
        if not candidates:
            raise HTTPException(500, "Failed to locate processed video output")

        produced = candidates[0]  # usually runs/out/<filename>.mp4

        # Move it to a stable temp path outside the context manager
        final_tmp = Path(TemporaryDirectory().name)  # a new temp folder
        final_tmp.mkdir(parents=True, exist_ok=True)
        final_path = final_tmp / "processed_video.mp4"
        shutil.move(str(produced), final_path)

    # Return file; tmp dir holding final_path lives until process cleans it
    return FileResponse(
        str(final_path),
        media_type="video/mp4",
        filename="processed_video.mp4",
        headers={"Content-Disposition": "attachment; filename=processed_video.mp4"},
    )
