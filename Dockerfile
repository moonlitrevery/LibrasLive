# LibrasLive Dockerfile
# Multi-stage build for optimized production deployment

# Stage 1: Base Python environment
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base as python-deps

# Create working directory
WORKDIR /app

# Copy Python requirements
COPY backend/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 3: Application
FROM python-deps as application

# Create app user for security
RUN adduser --disabled-password --gecos '' --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy backend files
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY data/ /app/data/
COPY notebooks/ /app/notebooks/
COPY README.md /app/

# Create required directories
RUN mkdir -p /app/backend/models \
    /app/backend/temp_audio \
    /app/data/landmarks \
    && chown -R appuser:appuser /app

# Copy startup script
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Switch to app user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# Expose ports
EXPOSE 5000 8000

# Default command
CMD ["/app/start.sh"]