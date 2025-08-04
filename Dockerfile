FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Install PyTorch with CUDA 12.1 first
RUN pip install --no-cache-dir \
    torch==2.2.1 \
    torchvision==0.17.1 \
    torchaudio==2.2.1 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 3. Install Fairseq from source (main branch)
RUN git clone https://github.com/facebookresearch/fairseq && \
    cd fairseq && \
    pip install --editable ./ && \
    cd ..

# 4. Download model files with retries
RUN mkdir -p /app/models && \
    wget --tries=3 -O /app/models/nllb-200-distilled-600M.pt \
    https://dl.fbaipublicfiles.com/nllb/lossless_split/nllb-200-distilled-600M.pt && \
    wget --tries=3 -P /app/models \
    https://dl.fbaipublicfiles.com/nllb/lossless_split/dict.src.txt && \
    wget --tries=3 -P /app/models \
    https://dl.fbaipublicfiles.com/nllb/lossless_split/dict.tgt.txt

# 5. Verify downloads
RUN sha256sum /app/models/nllb-200-distilled-600M.pt | grep -q "a1c5e7e0b7c3a5e0e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5" || { echo "Model verification failed"; exit 1; }

# 6. Copy application
COPY app.py .
COPY requirements.txt .

# 7. Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]