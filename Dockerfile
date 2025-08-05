FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------
# Step 1: System packages
# ---------------------
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils \
    python3-pip build-essential git curl wget \
    libsndfile1 libffi-dev libprotobuf-dev protobuf-compiler \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ---------------------
# Step 2: Python alias
# ---------------------
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python -m pip install --upgrade pip

# ---------------------
# Step 3: Install PyTorch with CUDA support (compatible version)
# ---------------------
RUN pip install torch==2.2.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# ---------------------
# Step 4: Install other dependencies
# ---------------------
RUN pip install ninja cython pybind11 numpy

# ---------------------
# Step 5: Clone and install fairseq (with compatible versions)
# ---------------------
WORKDIR /workspace
RUN git clone --recursive https://github.com/facebookresearch/fairseq.git

WORKDIR /workspace/fairseq
RUN pip install --no-build-isolation --editable .

# ---------------------
# Step 6: App Setup
# ---------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]

