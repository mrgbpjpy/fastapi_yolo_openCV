# ---- Base image (CPU) ----
FROM python:3.10-slim

# Fast, deterministic builds
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # keep the instance responsive on small CPUs
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    # prevent Ultralytics from pip-installing at runtime
    ULTRALYTICS_NOAUTOINSTALL=1 \
    # default single worker; Railway will set $PORT
    UVICORN_WORKERS=1

WORKDIR /app

# System libs needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (better layer cache)
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Railway will set PORT; still useful locally
EXPOSE 8000

# Use shell form so $PORT expands on Railway
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS}"]
