# Start from the CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

# ---------------------
# Step 1: System Setup
# ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    build-essential \
    git \
    curl \
    wget \
    libsndfile1 \
    libffi-dev \
    libprotobuf-dev \
    protobuf-compiler \
    cmake \
    ninja-build \
    gcc \
    g++ \
    libjpeg-dev \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# ---------------------
# Step 2: Python Setup
# ---------------------
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python -m pip install --upgrade pip && \
    pip install pip==24.0

# ---------------------
# Step 3: Install PyTorch
# ---------------------
RUN pip install --no-cache-dir torch==1.13.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# ---------------------
# Step 4: Install Fairseq
# ---------------------
RUN pip install --no-cache-dir fairseq==0.12.2

# ---------------------
# Step 5: Download Model
# ---------------------
RUN mkdir -p /app/models && \
    cd /app/models && \
    curl -L https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.joined-dict.ensemble.tar.gz -o wmt19.en-ru.tar.gz && \
    tar -xzvf wmt19.en-ru.tar.gz && \
    rm wmt19.en-ru.tar.gz && \
    mv wmt19.en-ru.joined-dict.ensemble en-ru

# ---------------------
# Step 6: Cleanup
# ---------------------
RUN apt-get purge -y build-essential gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# ---------------------
# Step 7: App Setup
# ---------------------
WORKDIR /app
COPY app.py .

# Entrypoint
CMD ["python", "app.py"]
