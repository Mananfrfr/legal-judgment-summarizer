FROM python:3.12-slim

# Install system dependencies (including tesseract for optional OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies with CPU-optimized PyTorch
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-app.txt

# Copy project files
COPY . .

# Pre-download and verify frozen model weights during image build
RUN python scripts/download_models.py

# Set environment defaults
ENV PYTHONPATH=/app/src
ENV APP_HOST=0.0.0.0
ENV APP_PORT=7861
ENV APP_ALLOW_UNAUTHENTICATED=true
ENV LEGAL_CPU_THREADS=2
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 7861

CMD ["python", "src/stage44_app.py"]
