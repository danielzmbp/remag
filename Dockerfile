# Multi-stage build for REMAG
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY pyproject.toml README.md ./
COPY remag ./remag

# Build the package and download all dependencies
RUN pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip wheel --no-cache-dir --wheel-dir=/wheels ./dist/*.whl && \
    pip wheel --no-cache-dir --wheel-dir=/wheels torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Final stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    samtools \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash remag

# Set working directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /wheels /wheels
COPY --from=builder /app/dist/*.whl /wheels/

# Install REMAG and all dependencies from wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels remag && \
    rm -rf /wheels

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