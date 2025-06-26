#!/bin/bash
set -e

echo "🚀 Starting AI Model Server..."
echo "📍 Working directory: $(pwd)"
echo "👤 Running as user: $(whoami)"
echo "🔧 User ID: $(id)"
echo "🐍 Python version: $(python3 --version)"
echo "🔧 CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'false')"

# Verify directory permissions
echo "📁 Checking directory permissions..."
ls -la /app/

# Create directories if they do not exist and set permissions
echo "📂 Setting up directories..."
mkdir -p models tmp logs config
chmod 755 models tmp logs config

# Test write permissions
echo "✅ Testing write permissions..."
echo "test" > logs/permission_test.log && rm logs/permission_test.log
echo "✅ Log directory permissions verified"

# Check disk space
echo "💾 Disk space:"
df -h /app

# Check memory
echo "🧠 Memory info:"
free -h

# Download models if needed
echo "📥 Checking and downloading models..."
python3 main.py --download-only || echo "⚠️ Model download failed, continuing with on-demand loading"

# Start the main application
echo "🌐 Starting Flask server..."
exec python3 main.py --host 0.0.0.0 --port 5000