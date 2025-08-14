FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    ULTRALYTICS_NOAUTOINSTALL=1 \
    UVICORN_WORKERS=1

WORKDIR /app

# Minimal libs for OpenCV runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy reqs early for caching
COPY requirements.txt .

# Prove Python & pip versions in the build log
RUN echo "=== PYTHON/PIP VERSIONS ===" \
 && python -V && pip -V \
 && python -m pip install --upgrade pip setuptools wheel

# Show exactly which requirements are being used
RUN echo "=== REQUIREMENTS CONTENTS ===" && cat requirements.txt

# Install Torch + Torchvision first from CPU wheel index (clearer errors if they fail)
RUN echo "=== INSTALL TORCH/TV (CPU) ===" \
 && pip install -v --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.1+cpu \
      torchvision==0.18.1+cpu

# Install the rest (verbose)
RUN echo "=== INSTALL REST OF REQUIREMENTS ===" \
 && pip install -v --no-cache-dir -r requirements.txt \
 && pip cache purge

# Copy the app
COPY . .

EXPOSE 8000
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS}"]
