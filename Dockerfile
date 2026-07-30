FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow/OpenCV if wheels need them
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend code
COPY backend/ backend/
COPY pathfinder.py .

# Frontend build (pre-built via Vercel, embedded for same-origin fallback)
COPY frontend/dist/ frontend/dist/

EXPOSE 7860

CMD ["python", "backend/Odin.py", "--host", "0.0.0.0", "--port", "7860"]
