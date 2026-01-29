# 🐳 Dockerfile - INE/IFE OCR API 🇲🇽🪪🤖
# ------------------------------------------------------------
# ✅ Corre Flask (WSGI) con Gunicorn 🦄
# ✅ Incluye libs del sistema necesarias para OpenCV / PaddleOCR
# ✅ Descarga modelos la primera vez (se recomienda volumen para cache)
#
# ⚠️ Nota sobre arquitecturas:
# - En Linux x86_64 (amd64) funciona directo.
# - En Apple Silicon (M1/M2) puede funcionar en arm64 si hay wheel,
#   pero si tienes problemas, usa docker compose con:
#   platform: linux/amd64  (más abajo te lo dejo comentado)
# ------------------------------------------------------------

FROM python:3.12-slim

# 🧠 Mejoras generales
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 🧰 Dependencias del sistema (OpenCV + OCR + performance)
# - libgl1 / libglib2.0-0: requeridas por OpenCV en muchos casos
# - libgomp1: OpenMP (paddle / numpy / etc.)
# - curl: útil para healthchecks/debug
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
  && rm -rf /var/lib/apt/lists/*

# 📁 App
WORKDIR /app

# 📦 Copiamos requirements primero para aprovechar cache
COPY requirements.txt /app/requirements.txt

# ⚠️ IMPORTANTE:
# Si tu requirements.txt es muy grande, el build tardará.
# Aun así, es lo más portable para correr en cualquier sistema.
RUN pip install --upgrade pip && pip install -r requirements.txt

# 📄 Copiamos el resto del proyecto
COPY . /app

# 🧾 Carpeta para caches de modelos (se recomienda mapear a volumen)
# PaddleOCR/PaddleX suelen guardar en /root/.paddleocr /root/.paddlex
RUN mkdir -p /root/.paddleocr /root/.paddlex

# 🔐 Permisos del entrypoint (si existe)
RUN chmod +x /app/entrypoint.sh

# 🌐 Puerto
EXPOSE 5001

# 🦄 Ejecutamos con Gunicorn (prod)
CMD ["/app/entrypoint.sh"]
