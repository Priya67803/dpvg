FROM python:3.12-slim

WORKDIR /app

# Install build/runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Environment defaults for container deployment:
# - Connect to host Ollama via host.docker.internal
# - Bind server to all interfaces (0.0.0.0:8000)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DPVG_OLLAMA_URL=http://host.docker.internal:11434 \
    DOCKER_CONTAINER=true \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "run.py"]
