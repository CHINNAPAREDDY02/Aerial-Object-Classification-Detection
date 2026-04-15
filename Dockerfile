# ============================================================
# Dockerfile — Aerial Object Classification & Detection
# ============================================================
# Build:   docker build -t aerial-classifier .
# Run app: docker run -p 8501:8501 aerial-classifier
# Run CLI: docker run aerial-classifier python scripts/predict_cli.py --help

FROM python:3.10-slim

# Labels
LABEL name="aerial-object-classifier"
LABEL description="Bird vs Drone Aerial Classification & Detection"

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker cache layer optimization)
COPY requirements.txt .

# Install Python dependencies (CPU-only TensorFlow for smaller image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        tensorflow-cpu>=2.12.0 \
        ultralytics>=8.0.0 \
        streamlit>=1.28.0 \
        opencv-python-headless>=4.7.0 \
        Pillow>=9.5.0 \
        numpy>=1.23.0 \
        pandas>=1.5.0 \
        scikit-learn>=1.2.0 \
        matplotlib>=3.7.0 \
        seaborn>=0.12.0 \
        PyYAML>=6.0 \
        tqdm>=4.65.0

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p models/saved models/checkpoints results/plots results/reports

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: launch Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
