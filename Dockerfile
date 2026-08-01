# Stage 1: Build the React frontend SPA
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
# Ensure public directory exists for Vite static asset bundling
RUN mkdir -p public
RUN npm run build

# Stage 2: Python backend runtime environment
FROM python:3.11-slim

# System dependencies for Pillow/OpenCV + Git & Git LFS for binary assets
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev libglib2.0-0 git git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Hugging Face Spaces requires running under user UID 1000
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository files including backend and git assets
COPY . .

# Pull Git LFS binary files (e.g. path.png, map images) so pointers are converted to real images
RUN git lfs pull || true

# Copy built frontend assets from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Ensure user 1000 owns the app directory for runtime state and checkpoint writing
RUN chown -R user:user $HOME/app

USER user

EXPOSE 7860

CMD ["python", "backend/Odin.py", "--host", "0.0.0.0", "--port", "7860"]
