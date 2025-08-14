FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    ULTRALYTICS_NOAUTOINSTALL=1 \
    UVICORN_WORKERS=1

WORKDIR /app

# OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy reqs early to leverage layer cache
COPY requirements.txt .

# Show exact versions (helps confirm Py 3.10)
RUN python -V && pip -V && python -m pip install --upgrade pip setuptools wheel

# Install Torch/TV first from the CPU index (clearer errors if it fails)
RUN pip install -v --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu

# Install the rest (verbose to surface the *first* real error)
RUN pip install -v --no-cache-dir -r requirements.txt && pip cache purge

# App files
COPY . .

EXPOSE 8000
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS}"]
