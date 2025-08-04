FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# 1. Install system dependencies with clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up workspace
WORKDIR /app

# 3. Install Python packages with explicit versions
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Download model files with retries and validation
RUN mkdir -p /app/models && \
    wget --tries=3 --waitretry=30 --retry-connrefused \
    -O /app/models/nllb-200-distilled-600M.pt \
    https://dl.fbaipublicfiles.com/nllb/lossless_split/nllb-200-distilled-600M.pt && \
    wget --tries=3 \
    -P /app/models \
    https://dl.fbaipublicfiles.com/nllb/lossless_split/dict.src.txt && \
    wget --tries=3 \
    -P /app/models \
    https://dl.fbaipublicfiles.com/nllb/lossless_split/dict.tgt.txt

# 5. Copy application
COPY app.py .

# 6. Environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

CMD ["python3", "app.py"]