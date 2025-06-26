# Dockerfile for AI Model Server
# Multi-stage build for optimized production image

# ==================== BUILD STAGE ====================
# เปลี่ยน Base Image เป็น NVIDIA CUDA image ที่มี Python ในตัว
# ให้ตรงกับ CUDA_VERSION ที่คุณต้องการใช้ (เช่น cu118 = CUDA 11.8)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder 

# Set build arguments (ARG CUDA_VERSION ไม่จำเป็นแล้วถ้า Base Image ตรง)
ARG PYTORCH_VERSION=2.0.0 # ตรวจสอบเวอร์ชัน PyTorch ใน requirements.txt ด้วย

# Install system dependencies for building
# build-essential และ cmake มีอยู่แล้วใน builder stage ซึ่งถูกต้อง
RUN apt-get update && apt-get install -y \
    python3 \             
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    # ไม่ต้องระบุ --index-url เมื่อใช้ NVIDIA CUDA base image
    pip3 install --no-cache-dir torch torchvision torchaudio && \ 
    pip3 install --no-cache-dir -r requirements.txt

# ==================== PRODUCTION STAGE ====================
# ใช้ NVIDIA CUDA image เดียวกันกับ builder stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS production 

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    CUDA_VISIBLE_DEVICES="" \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TRANSFORMERS_OFFLINE=0 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MALLOC_CONF=background_thread:true,metadata_thp:auto \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Install runtime system dependencies (เพิ่ม build-essential และ cmake ที่นี่)
RUN apt-get update && apt-get install -y \
    python3 \             
    python3-pip \
    build-essential \ 
    cmake \           
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    ffmpeg \
    poppler-utils \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean 

# Create non-root user FIRST
RUN useradd --create-home --shell /bin/bash --uid 1000 app

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local /usr/local

# Create necessary directories and set ownership BEFORE switching user
RUN mkdir -p /app/models /app/tmp /app/logs /app/config && \
    chmod 755 /app && \
    chmod 755 /app/models /app/tmp /app/logs /app/config && \
    chown -R app:app /app

# Copy application files with proper ownership
COPY --chown=app:app app.py main.py requirements.txt ./
COPY --chown=app:app utils/ ./utils/

# Copy and prepare startup script
COPY --chown=app:app start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Switch to non-root user AFTER setting up permissions
USER app

# Create a test log file to verify permissions
RUN touch /app/logs/test.log && rm /app/logs/test.log

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Volume mounts for persistent data
VOLUME ["/app/models", "/app/logs", "/app/tmp"]

# Default command
CMD ["/app/start.sh"]

# ==================== LABELS ====================
LABEL maintainer="PnTnk" \
      version="1.0.0" \
      description="AI Model Server with multimodal capabilities" \
      org.opencontainers.image.title="AI Model Server" \
      org.opencontainers.image.description="Flask API server for AI models with text, vision, audio, and video processing" \
      org.opencontainers.image.vendor="Your Organization" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/PnTnk/ai-model-server.git"