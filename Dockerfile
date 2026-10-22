FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} appuser \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m appuser

COPY app/ ./app/
COPY config/ ./config/

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Default: web server. CLI available via: python -m app <command>
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
