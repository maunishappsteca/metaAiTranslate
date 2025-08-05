# Use PyTorch image with CUDA (works for both CPU and GPU)
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime


# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch fairseq flask runpod

# Create model directory
RUN mkdir -p /workspace/models/en-ru

# Download pretrained WMT19 EN-RU model
RUN wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz -P /workspace/models/en-ru && \
    tar -xvf /workspace/models/en-ru/wmt19.en-ru.ensemble.tar.gz -C /workspace/models/en-ru

# Copy app code
COPY app.py /workspace/app.py

# Set working directory
WORKDIR /workspace

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
