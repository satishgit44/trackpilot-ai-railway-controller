# TrackPilot AI Railway Controller - Multi-stage Docker build
# This Dockerfile creates a production-ready container with all components

# ================================
# Stage 1: LibTorch and C++ Core Build
# ================================
FROM ubuntu:22.04 AS cpp-builder

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    pkg-config \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-filesystem-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install LibTorch
WORKDIR /tmp
RUN wget -q https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip \
    && unzip -q libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip \
    && mv libtorch /opt/libtorch \
    && rm libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip

# Build C++ core
COPY core/ /src/core/
WORKDIR /src/core

RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/libtorch \
    -DBUILD_TESTING=OFF \
    && cmake --build build --config Release --parallel $(nproc) \
    && cmake --install build --prefix /opt/trackpilot

# ================================
# Stage 2: Python AI Training Environment
# ================================
FROM python:3.10-slim AS python-builder

# Install system dependencies for Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY python/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Copy Python source code
COPY python/ /src/python/
WORKDIR /src/python

# Pre-compile Python bytecode
RUN python -m compileall .

# ================================
# Stage 3: Node.js Frontend Build
# ================================
FROM node:18-alpine AS frontend-builder

# Set working directory
WORKDIR /src/web

# Copy package files and install dependencies
COPY web/package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy source code and build
COPY web/ .
RUN npm run build

# ================================
# Stage 4: Production Runtime
# ================================
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libboost-system1.74.0 \
    libboost-thread1.74.0 \
    libboost-filesystem1.74.0 \
    libssl3 \
    python3 \
    python3-pip \
    nodejs \
    npm \
    nginx \
    supervisor \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -r -s /bin/bash -d /app trackpilot

# Copy LibTorch libraries
COPY --from=cpp-builder /opt/libtorch/lib/ /usr/local/lib/
RUN ldconfig

# Copy C++ core binary and configs
COPY --from=cpp-builder /opt/trackpilot/ /app/core/
RUN chmod +x /app/core/bin/trackpilot_core

# Copy Python environment and source
COPY --from=python-builder /opt/venv/ /app/python/venv/
COPY --from=python-builder /src/python/ /app/python/
ENV PATH="/app/python/venv/bin:$PATH"

# Copy frontend build
COPY --from=frontend-builder /src/web/dist/ /app/web/dist/

# Create necessary directories
RUN mkdir -p /app/{logs,models,config,data} \
    && chown -R trackpilot:trackpilot /app

# Copy configuration files
COPY infra/config/ /app/config/
COPY infra/nginx/ /etc/nginx/
COPY infra/supervisor/ /etc/supervisor/

# Create default configuration
WORKDIR /app
COPY infra/scripts/create-config.sh /tmp/
RUN chmod +x /tmp/create-config.sh && /tmp/create-config.sh

# Health check script
COPY infra/scripts/healthcheck.sh /app/
RUN chmod +x /app/healthcheck.sh

# Expose ports
EXPOSE 80 8080 3000

# Set up volumes for persistent data
VOLUME ["/app/logs", "/app/models", "/app/data"]

# Switch to application user
USER trackpilot

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Default command
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]

# ================================
# Development Stage (optional)
# ================================
FROM runtime AS development

# Switch back to root for development tools
USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    gdb \
    valgrind \
    strace \
    tcpdump \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Python development dependencies
RUN /app/python/venv/bin/pip install \
    jupyter \
    ipython \
    black \
    flake8 \
    pytest \
    pytest-cov

# Development configuration overrides
COPY infra/config/development.json /app/config/config.json

# Switch back to application user
USER trackpilot

# Development command with hot reload
CMD ["supervisord", "-c", "/etc/supervisor/supervisord-dev.conf"]

# ================================
# Build Arguments and Labels
# ================================
ARG BUILD_VERSION=1.0.0
ARG BUILD_DATE
ARG VCS_REF

LABEL org.opencontainers.image.title="TrackPilot AI Railway Controller" \
      org.opencontainers.image.description="Hybrid AI system for real-time train scheduling with human oversight" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.authors="Team Neuronauts" \
      org.opencontainers.image.url="https://github.com/satishgit44/trackpilot-ai-railway-controller" \
      org.opencontainers.image.source="https://github.com/satishgit44/trackpilot-ai-railway-controller.git" \
      org.opencontainers.image.licenses="MIT"