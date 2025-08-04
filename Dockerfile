FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils \
    python3-pip \
    build-essential \
    git wget curl \
    libsndfile1 \
    libffi-dev \
    libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Set python alias
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Install ninja before building fairseq
RUN pip install ninja

# Clone fairseq with submodules (IMPORTANT)
RUN git clone --recursive https://github.com/facebookresearch/fairseq.git && \
    cd fairseq && \
    pip install --editable . && \
    cd ..

# Set working directory
WORKDIR /app

# Install other Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
