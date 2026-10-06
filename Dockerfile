FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY frontend ./frontend
COPY monitoring ./monitoring
COPY deployment ./deployment

EXPOSE 8001

CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8001}"]