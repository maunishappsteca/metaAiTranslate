FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up workspace
WORKDIR /app

# 3. Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Download model files with multiple fallback sources
RUN mkdir -p /app/models && \
    (wget --tries=3 -O /app/models/nllb-200-distilled-600M.pt \
    https://dl.fbaipublicfiles.com/nllb/lossless_split/nllb-200-distilled-600M.pt || \
    curl -L -o /app/models/nllb-200-distilled-600M.pt \
    https://storage.googleapis.com/ai2-mosaic-public/projects/nllb/nllb-200-distilled-600M.pt) && \
    wget --tries=3 -P /app/models \
    https://dl.fbaipublicfiles.com/nllb/lossless_split/dict.src.txt && \
    wget --tries=3 -P /app/models \
    https://dl.fbaipublicfiles.com/nllb/lossless_split/dict.tgt.txt

# 5. Verify downloads
RUN ls -lh /app/models/ && \
    [ -s /app/models/nllb-200-distilled-600M.pt ] || { echo "Model download failed"; exit 1; }

# 6. Copy application
COPY app.py .

CMD ["python3", "app.py"]