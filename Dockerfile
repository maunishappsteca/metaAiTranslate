# Use official Python image with necessary system packages
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    libsndfile1 \
    libffi-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (torch first, then others)
RUN pip install --upgrade pip

# Install torch first (with CUDA 11.6 support) - adjust if CPU only
RUN pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# Install fairseq, flask, runpod
RUN pip install fairseq flask runpod

# Copy app files
COPY app.py .

# Default command
CMD ["python", "app.py"]
