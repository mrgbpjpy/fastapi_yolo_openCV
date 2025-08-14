# FastAPI official documentation: https://fastapi.tiangolo.com/
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
# FastAPI FileResponse documentation: https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse
from fastapi.responses import FileResponse
# Python os module documentation: https://docs.python.org/3/library/os.html
import os
# Python tempfile module documentation: https://docs.python.org/3/library/tempfile.html
from tempfile import NamedTemporaryFile
# OpenCV documentation: https://docs.opencv.org/4.x/
import cv2
# Ultralytics YOLO documentation: https://docs.ultralytics.com/
from ultralytics import YOLO
# CORS middleware: https://fastapi.tiangolo.com/tutorial/cors/
from fastapi.middleware.cors import CORSMiddleware

# FastAPI app instantiation: https://fastapi.tiangolo.com/tutorial/first-steps/
app = FastAPI()

# Add CORS middleware: https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://7ddd95.csb.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model: https://docs.ultralytics.com/reference/engine/model/#ultralytics.engine.model.YOLO.__init__
model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 nano model

# Function to remove file: Custom utility, no external doc
def remove_file(path: str):
    """Background task to clean up temporary files after response."""
    # os.remove documentation: https://docs.python.org/3/library/os.html#os.remove
    os.remove(path)

# FastAPI POST endpoint: https://fastapi.tiangolo.com/tutorial/body/#request-body
@app.post("/upload_video")
def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Validate file extension: Custom logic
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        # HTTPException documentation: https://fastapi.tiangolo.com/tutorial/handling-errors/#use-httpexception
        raise HTTPException(status_code=400, detail="Invalid video file format")

    # Create temporary file for input: https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
    input_temp = NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        # Read file contents: https://fastapi.tiangolo.com/tutorial/request-files/#uploadfile
        contents = file.file.read()
        # Write to temp file: Python file write https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
        input_temp.write(contents)
        # Close temp file: Python file close https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
        input_temp.close()
        input_path = input_temp.name
    except Exception as e:
        # Print error: Python print https://docs.python.org/3/library/functions.html#print
        print(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Error saving uploaded file")
    finally:
        # Close UploadFile: https://fastapi.tiangolo.com/tutorial/request-files/#uploadfile
        file.file.close()

    # Open video capture with FFmpeg backend: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
    # CAP_FFMPEG constant: https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html
    cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
    # Check if opened: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html#a9ac7f4b1cdfe62486e15318e1a1a0c44
    if not cap.isOpened():
        os.remove(input_path)
        raise HTTPException(status_code=500, detail="Error opening video file (even with FFmpeg)")

    # Get video properties: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html#a5d5f5dacb77e520a0b30a7b3cb17730a
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Print video info: Python print https://docs.python.org/3/library/functions.html#print
    print(f"Input video: {width}x{height} @ {fps} FPS, estimated {total_frames} frames")

    if width == 0 or height == 0:
        # Release capture: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html#a9d2c2978e23696b916efe1ad185be7b8
        cap.release()
        os.remove(input_path)
        raise HTTPException(status_code=500, detail="Invalid video dimensions - codec/backend issue likely")

    # Create temporary file for output: https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
    output_temp = NamedTemporaryFile(delete=False, suffix=".mp4")
    output_temp.close()
    output_path = output_temp.name

    # FourCC for H264: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    # VideoWriter: https://docs.opencv.org/4.x/de/da9/classcv_1_1VideoWriter.html
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Check if writer opened: https://docs.opencv.org/4.x/de/da9/classcv_1_1VideoWriter.html#a0a48d6f52d38b2b1d333c7e7671ff05b
    if not out.isOpened():
        cap.release()
        os.remove(input_path)
        raise HTTPException(status_code=500, detail="Error initializing video writer - check FFmpeg codec support")

    frame_count = 0
    try:
        # Loop through frames
        while cap.isOpened():
            # Read frame: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html#a57c0e81e83e60f36c83027dc2a1885b1
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO track: https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.track
            results = model.track(frame, persist=True)

            # Plot annotations: https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.plot
            annotated_frame = results[0].plot()

            # Write frame: https://docs.opencv.org/4.x/de/da9/classcv_1_1VideoWriter.html#a25c3df94779dcbd5c9744b399f2e1b32
            out.write(annotated_frame)
            frame_count += 1

        if frame_count == 0:
            raise Exception("No frames could be read from the input video - FFmpeg may not support this MP4's codec/profile")

        print(f"Processed and wrote {frame_count} frames successfully")

    except Exception as e:
        print(f"Error during video processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        # Release resources
        cap.release()
        # Release writer: https://docs.opencv.org/4.x/de/da9/classcv_1_1VideoWriter.html#a29b870e50e527c2056529a2c35d9d612
        out.release()
        os.remove(input_path)  # Clean up input

    # Add background task: https://fastapi.tiangolo.com/tutorial/background-tasks/
    background_tasks.add_task(remove_file, output_path)
    # Return FileResponse: https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="processed_video.mp4",
        headers={"Content-Disposition": "attachment; filename=processed_video.mp4"}
    )
