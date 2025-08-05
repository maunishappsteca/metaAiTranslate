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
# Step 5: Download and Extract Model
# ---------------------
# Added 'set -euxo pipefail' for better debugging. This will make the script
# exit immediately if any command fails and print the failing command.
# The curl command downloads the file, tar extracts it, rm removes the archive,
# and mv renames the extracted directory.


# Create the models directory
RUN set -euxo pipefail; \
    mkdir -p /app/models

# Change to the models directory for easier path handling
WORKDIR /app/models

# Download the model archive
RUN set -euxo pipefail; \
    echo "Downloading model..." && \
    curl -L -f -o wmt19.en-ru.tar.gz https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.joined-dict.ensemble.tar.gz && \
    echo "Download complete. Listing contents of /app/models:" && \
    ls -l

# Extract the model archive
RUN set -euxo pipefail; \
    echo "Extracting model..." && \
    tar -xzvf wmt19.en-ru.tar.gz && \
    echo "Extraction complete. Listing contents of /app/models:" && \
    ls -l

# Rename the extracted directory
RUN set -euxo pipefail; \
    echo "Renaming extracted directory..." && \
    mv wmt19.en-ru.joined-dict.ensemble en-ru && \
    echo "Renaming complete. Listing contents of /app/models:" && \
    ls -l

# Remove the downloaded archive to save space
RUN set -euxo pipefail; \
    echo "Removing temporary archive..." && \
    rm wmt19.en-ru.tar.gz && \
    echo "Cleanup complete. Final contents of /app/models:" && \
    ls -l
    

# ---------------------
# Step 6: Cleanup
# ---------------------
# Purge build dependencies to reduce final image size.
RUN apt-get purge -y build-essential gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# ---------------------
# Step 7: App Setup
# ---------------------
WORKDIR /app
COPY app.py .

# Entrypoint for the application
CMD ["python", "app.py"]
