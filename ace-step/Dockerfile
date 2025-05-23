ARG DOCKER_FROM=nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Base NVidia CUDA Ubuntu image
FROM $DOCKER_FROM AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7865 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user to run the application
RUN useradd -m -u 1001 appuser

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/ace-step/ACE-Step.git .

# Install specific PyTorch version compatible with CUDA 12.8
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir hf_transfer peft && \
    pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --no-deps && \
    pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --no-deps --ignore-installed torch torchaudio torchvision
RUN pip3 install --no-cache-dir .

# Copy the download script
COPY download_model.py /app/download_model.py
RUN chmod +x /app/download_model.py

# Create the home directory for appuser and ensure it has the right permissions
RUN mkdir -p /home/appuser && chown -R appuser:appuser /home/appuser

# Create matplotlib cache directory with proper permissions
RUN mkdir -p /home/appuser/.cache/matplotlib && chown -R appuser:appuser /home/appuser/.cache

# Ensure target directories for volumes exist and have correct initial ownership
RUN mkdir -p /app/outputs /app/checkpoints /app/logs && \
    chown -R appuser:appuser /app/outputs /app/logs

# Download the model during build (this will cache the model in the Docker image)
# Run as root to ensure we can create the symbolic link
RUN python3 /app/download_model.py

# Change ownership of app files to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 7865

VOLUME [ "/app/checkpoints", "/app/outputs", "/app/logs" ]

# Set healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=5 \
  CMD curl -f http://localhost:7865/ || exit 1


# Command to run the application with GPU support
CMD ["python3", "acestep/gui.py", "--server_name", "0.0.0.0", "--bf16", "true", "--checkpoint_path", "/app/checkpoints"]
