# Multi-stage build for REMAG
ARG PYTHON_VERSION=3.11
ARG MINIPROT_COMMIT=671db243f964a68bd724af11cd9964d840f29c43

FROM python:${PYTHON_VERSION}-slim AS miniprot-builder

ARG MINIPROT_COMMIT

# Build miniprot from source because it is required at runtime and is not
# available in the base image package repositories.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN git clone --branch v0.18 --depth 1 https://github.com/lh3/miniprot.git && \
    test "$(git -C miniprot rev-parse HEAD)" = "${MINIPROT_COMMIT}" && \
    make -C miniprot

FROM python:${PYTHON_VERSION}-slim AS builder

# Set working directory
WORKDIR /app

# Copy only the files needed to build the wheel
COPY pyproject.toml README.md LICENSE ./
COPY remag ./remag

# Build the package
RUN pip install --no-cache-dir build && \
    python -m build --wheel

# Final stage
FROM python:${PYTHON_VERSION}-slim

# Install external runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    samtools \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash remag

# Copy the built wheel from builder
COPY --from=builder /app/dist/*.whl /tmp/

# Copy miniprot binary built from source
COPY --from=miniprot-builder /tmp/miniprot/miniprot /usr/local/bin/miniprot

# Install REMAG and all dependencies
# Install PyTorch CPU version first to avoid downloading large CUDA versions
# Let pip resolve versions based on pyproject.toml constraints
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl && \
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
