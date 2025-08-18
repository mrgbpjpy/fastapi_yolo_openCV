# =========================
# Stage 1: build dependencies
# =========================
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

# System libs needed to compile/install wheels and run OpenCV at build-time
# (no ffmpeg here; only the final image needs it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip toolchain
RUN python -m pip install --upgrade pip setuptools wheel

# Install Torch CPU wheels first (avoid pulling CUDA)
RUN pip install -v --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1+cpu torchvision==0.18.1+cpu

# Install the rest
RUN pip install -v --no-cache-dir -r requirements.txt


# =========================
# Stage 2: final runtime image
# =========================
FROM python:3.10-slim

# Important: FFmpeg/FFprobe for web-safe MP4 output
# Plus the minimal OpenCV/torch runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    ULTRALYTICS_NOAUTOINSTALL=1 \
    ULTRALYTICS_IGNORE_REQUIREMENTS=1 \
    UVICORN_WORKERS=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy site-packages and console scripts from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Your app code
COPY . .

# Railway exposes $PORT; default to 8000 locally
EXPOSE 8000

# Run FastAPI via Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS}"]
