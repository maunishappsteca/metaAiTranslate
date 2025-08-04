FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model files
RUN mkdir -p /app/models && \
    wget -O /app/models/nllb-200-distilled-600M.pt \
    https://dl.fbaipublicfiles.com/nllb/lossless_split/nllb-200-distilled-600M.pt && \
    wget -P /app/models https://dl.fbaipublicfiles.com/nllb/lossless_split/dictionary.txt

# Copy application code
COPY app.py .

CMD ["python3", "app.py"]