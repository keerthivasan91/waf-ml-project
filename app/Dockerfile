FROM python:3.11-slim

WORKDIR /app

# system deps for onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# copy trained models from ml/exported_models at build time
# (volume mount in docker-compose for dev, COPY for prod)
RUN mkdir -p /app/models

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]