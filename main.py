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
    import cv2  # lazy import keeps startup instant

    # --- light validation ---
    name = (file.filename or "video.mp4")
    lower = name.lower()
    if not lower.endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(400, "Invalid video file format")

    # --- temp IO setup ---
    from tempfile import TemporaryDirectory
    from pathlib import Path
    import shutil

    with TemporaryDirectory() as workdir:
        work = Path(workdir)
        in_dir = work / "input"
        in_dir.mkdir(parents=True, exist_ok=True)
        in_path = in_dir / name

        try:
            in_path.write_bytes(file.file.read())
        except Exception as e:
            raise HTTPException(500, f"Error saving upload: {e}")
        finally:
            try:
                file.file.close()
            except Exception:
                pass

        # --- open input to fetch video properties ---
        cap = cv2.VideoCapture(str(in_path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise HTTPException(500, "Error opening video file")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        if w <= 0 or h <= 0:
            raise HTTPException(500, "Invalid video dimensions")

        # --- writer (CPU-friendly mp4v) ---
        out_tmp_dir = Path(TemporaryDirectory().name)
        out_tmp_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_tmp_dir / "processed_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if not out.isOpened():
            raise HTTPException(500, "Error initializing video writer")

        # --- tracking (stream) ---
        model = get_model()
        frames = 0
        try:
            # stream=True prevents RAM buildup; persist=True keeps track IDs
            for r in model.track(
                source=str(in_path),
                stream=True,
                imgsz=320,               # speed knob (try 256 for faster)
                vid_stride=2,            # process every other frame
                conf=0.30,
                iou=0.45,
                device="cpu",
                tracker="bytetrack.yaml",  # avoids lapx
                persist=True,
                verbose=False,
            ):
                # r.orig_img is the original frame (BGR), r.plot() draws boxes/labels
                frame = r.plot()  # annotated with boxes & (sometimes) IDs

                # ---- ensure IDs are visible: draw them ourselves if necessary ----
                try:
                    ids = r.boxes.id  # tensor of track IDs or None
                    xyxy = r.boxes.xyxy  # (N, 4)
                    if ids is not None and xyxy is not None:
                        import numpy as np
                        ids_np = ids.cpu().numpy().astype(int)
                        boxes_np = xyxy.cpu().numpy().astype(int)
                        for (x1, y1, x2, y2), tid in zip(boxes_np, ids_np):
                            # label baseline above the box
                            label = f"ID {tid}"
                            # draw a filled background for readability
                            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 90, y1), (0, 0, 0), -1)
                            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                except Exception:
                    # if anything goes wrong, we still write the plotted frame
                    pass

                out.write(frame)
                frames += 1

            if frames == 0:
                raise RuntimeError("No frames processed (empty or unsupported video).")
        except Exception as e:
            raise HTTPException(500, f"Error during tracking: {e}")
        finally:
            out.release()

    # --- return the produced file ---
    return FileResponse(
        str(out_path),
        media_type="video/mp4",
        filename="processed_video.mp4",
        headers={"Content-Disposition": "attachment; filename=processed_video.mp4"},
    )
