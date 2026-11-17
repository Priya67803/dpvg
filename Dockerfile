FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HOST=0.0.0.0 PORT=8000 DPVG_OLLAMA_URL=http://host.docker.internal:11434
EXPOSE 8000
CMD ["python", "run.py"]
