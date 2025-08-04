FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    ninja-build \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install Python deps
RUN pip install --upgrade pip

# Clone and install fairseq manually
RUN git clone https://github.com/facebookresearch/fairseq.git && \
    cd fairseq && \
    pip install --editable ./ && \
    cd ..

# Copy your app code
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
