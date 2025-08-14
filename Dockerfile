FROM python:3.8-slim

  # Set working directory
  WORKDIR /app

  # Install system dependencies for OpenCV
  RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*

  # Copy requirements file
  COPY requirements.txt .

  # Install Python dependencies
  RUN pip install --no-cache-dir -r requirements.txt

  # Copy application code
  COPY . .

  # Expose port 8000
  EXPOSE 8000

  # Run Uvicorn
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
