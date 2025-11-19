FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies for GGUF inference
RUN pip install --no-cache-dir \
    llama-cpp-python>=0.2.0 \
    huggingface_hub>=0.20.0 \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.0.0 \
    requests

# Copy API server script
COPY scripts/gguf_api_server.py /app/api_server.py

# Set environment variables for CPU
ENV CUDA_VISIBLE_DEVICES=""

# Expose API port
EXPOSE 8000

# Default command
CMD ["python3", "/app/api_server.py"]

