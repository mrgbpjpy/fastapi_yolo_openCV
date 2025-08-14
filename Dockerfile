FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    ULTRALYTICS_NOAUTOINSTALL=1 \
    UVICORN_WORKERS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Prove versions; upgrade tooling
RUN echo "=== PYTHON/PIP ===" && python -V && pip -V \
 && python -m pip install --upgrade pip setuptools wheel

# Show reqs (so we know which file is used)
RUN echo "=== REQUIREMENTS ===" && cat requirements.txt

# Install Torch/TV CPU first (clearer if this step fails)
RUN echo "=== TORCH/TV CPU INSTALL ===" \
 && pip install -v --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.1+cpu \
      torchvision==0.18.1+cpu

# Install the rest (verbose for clear error)
RUN echo "=== INSTALL THE REST ===" \
 && pip install -v --no-cache-dir -r requirements.txt \
 && pip cache purge

COPY . .

EXPOSE 8000
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS}"]
