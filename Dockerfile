# Tek imaj; api, worker'lar ve gönderici aynı imajdan farklı komutla çalışır.
# Modeller imaja GİRMEZ: ./models klasörü salt-okunur bağlanır (imaj küçük kalır,
# internetsiz sunucuya models/ ayrıca taşınır, sha256'lar MANIFEST.json'da).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/models DATA_DIR=/data/uploads

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server]" && useradd --create-home --uid 1000 app \
    && mkdir -p /data/uploads && chown -R app /data

USER app
EXPOSE 8000
CMD ["uvicorn", "live_stream.interface.api:app", "--host", "0.0.0.0", "--port", "8000"]
