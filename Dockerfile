# Multi-stage build for REMAG
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code and git metadata for versioning
COPY pyproject.toml README.md ./
COPY remag ./remag
COPY .git ./.git

# Build the package
# If VERSION build arg is provided, use it as fallback for setuptools-scm
ARG VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_REMAG=${VERSION}
RUN pip install --no-cache-dir build && \
    python -m build --wheel

# Final stage
FROM python:3.9-slim

# Install runtime dependencies and build tools for packages that need compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    samtools \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash remag

# Copy the built wheel from builder
COPY --from=builder /app/dist/*.whl /tmp/

# Install REMAG and all dependencies
# Install PyTorch CPU version first to avoid downloading large CUDA versions
# Let pip resolve versions based on pyproject.toml constraints
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER remag

# Set environment variables
ENV PATH="/home/remag/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
# Set default log level to INFO to match pip version behavior
ENV LOG_LEVEL=INFO

# Create working directory for user
WORKDIR /data

# Entry point
ENTRYPOINT ["remag"]
CMD ["--help"]