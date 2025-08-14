FROM python:3.8-slim
WORKDIR /app

# OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Stop Ultralytics from pip-installing at runtime
ENV ULTRALYTICS_NOAUTOINSTALL=1
# Keep thread usage sane on small CPUs
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

COPY . .
EXPOSE 8000

# IMPORTANT: shell form so $PORT expands
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
