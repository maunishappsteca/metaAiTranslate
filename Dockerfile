# Start from the base image provided by RunPod
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variable to prevent interactive prompts during apt-get installs
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------
# Step 1: System packages and build tools
# ---------------------
# Install Python 3.10 and its development headers, pip, and essential build tools.
# libsndfile1 is required by fairseq for audio processing.
# libffi-dev, libprotobuf-dev, protobuf-compiler, cmake, ninja-build, gcc, g++
# are crucial for compiling Python packages that have C/C++ extensions (like fairseq).
# libjpeg-dev and zlib1g-dev are common image processing dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils \
    python3-pip build-essential git curl wget \
    libsndfile1 libffi-dev libprotobuf-dev protobuf-compiler \
    cmake ninja-build gcc g++ libjpeg-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# ---------------------
# Step 2: Python alias and pip upgrade
# ---------------------
# Create symbolic links for python and python3 to point to python3.10.
# Upgrade pip to ensure we have the latest version for installing packages.
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    python -m pip install --upgrade pip && \
    pip install pip==24.0

# ---------------------
# Step 3: Install Python packages
# ---------------------
# Install PyTorch first, as fairseq depends on it.
# We use the specific version 1.13.1+cu116 as in your original Dockerfile,
# which is compatible with fairseq's requirement of >= 1.10.0.
RUN pip install --no-cache-dir torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# Clone the fairseq repository and install it in editable mode.
# This approach directly follows the fairseq documentation's recommendation
# for local development, which is more robust for building complex libraries
# from source and ensures all necessary C/C++ components are compiled correctly.
# We checkout the specific v0.12.2 tag to match your desired version.
RUN git clone https://github.com/pytorch/fairseq.git /usr/local/src/fairseq_repo && \
    cd /usr/local/src/fairseq_repo && \
    git checkout v0.12.2 && \
    pip install --no-cache-dir --editable .

# Install other application-specific Python packages.
# Installing them after fairseq allows pip to resolve any potential
# shared dependencies more effectively.
RUN pip install --no-cache-dir flask runpod

# ---------------------
# Step 4: Download Fairseq EN→RU model
# ---------------------
# Create directory for models and download the pre-trained WMT19 EN->RU ensemble model.
# Extract the model files and then remove the tar.gz archive to save space.
RUN python -c "\
from fairseq import checkpoint_utils;\
checkpoint_utils.download_ensemble_and_task('wmt19.en-ru', save_dir='/app/models/en-ru')\
"


# ---------------------
# Step 5: App Setup
# ---------------------
# Copy your application script into the /app directory in the container.
COPY app.py /app/app.py

# Set the working directory for subsequent instructions and when the container starts.
WORKDIR /app

# Define the default command to run when the container starts.
CMD ["python", "app.py"]
