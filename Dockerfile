FROM python:3.11-slim

# System dependencies unstructured[pdf] needs for PDF parsing/OCR
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT — bind to it, not a hardcoded port
CMD uvicorn api:app --host 0.0.0.0 --port $PORT