#!/usr/bin/env sh
# 🦄 Entrypoint - Gunicorn launcher
# ---------------------------------
# ✅ Permite configurar workers/threads/timeout por variables de entorno
# ✅ Exponen /health para docker healthcheck

set -e

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-5001}"

GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"
GUNICORN_THREADS="${GUNICORN_THREADS:-4}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-120}"

echo "🪪 Starting INE OCR API with Gunicorn 🦄"
echo "🌐 Listening on: ${APP_HOST}:${APP_PORT}"
echo "⚙️  workers=${GUNICORN_WORKERS} threads=${GUNICORN_THREADS} timeout=${GUNICORN_TIMEOUT}s"

exec gunicorn \
  --bind "${APP_HOST}:${APP_PORT}" \
  --workers "${GUNICORN_WORKERS}" \
  --threads "${GUNICORN_THREADS}" \
  --timeout "${GUNICORN_TIMEOUT}" \
  --access-logfile - \
  --error-logfile - \
  main:app
