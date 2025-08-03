# Multi-stage build for REMAG
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY pyproject.toml README.md ./
COPY remag ./remag

# Build the package
RUN pip install --no-cache-dir build && \
    python -m build --wheel

# Final stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    samtools \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash remag

# Copy the built wheel from builder
COPY --from=builder /app/dist/*.whl /tmp/

# Install REMAG and all dependencies
# Install PyTorch CPU version first to avoid downloading large CUDA versions
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Switch to non-root user
USER remag

# Set environment variables
ENV PATH="/home/remag/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# Create working directory for user
WORKDIR /data

# Entry point
ENTRYPOINT ["remag"]
CMD ["--help"]