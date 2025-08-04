FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    # Required for pycld2
    libicu-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

CMD ["python3", "app.py"]