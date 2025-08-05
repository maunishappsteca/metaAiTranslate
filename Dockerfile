# Base image with CUDA (adjust if using CPU-only)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------
# Step 1: System packages
# ---------------------
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils \
    python3-pip build-essential curl wget \
    libsndfile1 libffi-dev libprotobuf-dev protobuf-compiler \
    cmake git && rm -rf /var/lib/apt/lists/*

# ---------------------
# Step 2: Python alias
# ---------------------
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python -m pip install --upgrade pip

# ---------------------
# Step 3: Install Python packages
# ---------------------
RUN pip install --no-cache-dir torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir fairseq flask runpod

# ---------------------
# Step 4: Download Fairseq EN→RU model
# ---------------------
RUN mkdir -p /app/models/en-ru && \
    wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz -P /app/models/en-ru && \
    tar -xvf /app/models/en-ru/wmt19.en-ru.ensemble.tar.gz -C /app/models/en-ru

# ---------------------
# Step 5: App Setup
# ---------------------
COPY app.py /app/app.py

WORKDIR /app

# RunPod Serverless starts from this
CMD ["python", "app.py"]
