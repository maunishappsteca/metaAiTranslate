# Base image with CUDA 12.1 and Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    libsndfile1 \
    ninja-build \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies (except fairseq)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install fairseq from source in editable mode
RUN git clone https://github.com/facebookresearch/fairseq.git && \
    cd fairseq && \
    pip install --editable . && \
    cd ..

# (Optional) Install NVIDIA apex for faster training
# RUN git clone https://github.com/NVIDIA/apex.git && \
#     cd apex && \
#     pip install --no-cache-dir -v --disable-pip-version-check \
#     --global-option="--cpp_ext" \
#     --global-option="--cuda_ext" \
#     --global-option="--deprecated_fused_adam" \
#     --global-option="--xentropy" \
#     --global-option="--fast_multihead_attn" \
#     . && cd ..

# Set working directory
WORKDIR /app
COPY . .

# Default run command
CMD ["python", "app.py"]