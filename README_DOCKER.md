# 🐳 Docker Setup - INE/IFE OCR API 🇲🇽🪪🤖

Este paquete agrega los archivos necesarios para correr tu API en **cualquier sistema** usando Docker:

✅ Dockerfile  
✅ docker-compose.yml  
✅ .dockerignore  
✅ entrypoint.sh (Gunicorn)  

---

## 1) Archivos incluidos 📁

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `entrypoint.sh`

---

## 2) Requisito: endpoint /health ❤️

Para que el healthcheck funcione, agrega esto a tu `main.py` (si aún no lo tienes):

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

📌 Si tu Flask no tiene `.get`, usa `.route`:

```python
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
```

---

## 3) Build + Run 🚀

Desde la carpeta del proyecto (donde está `Dockerfile`):

```bash
docker compose up --build
```

Abre:
- API: `http://localhost:5001`
- Swagger: `http://localhost:5001/apidocs/`
- Health: `http://localhost:5001/health`

---

## 4) Apple Silicon (M1/M2) 🍎⚠️

Si Paddle/PaddleOCR te falla por arquitectura, edita `docker-compose.yml` y descomenta:

```yaml
platform: linux/amd64
```

Eso corre con emulación y suele funcionar mejor para algunas wheels.

---

## 5) Persistencia de modelos 🧠

Se crean volúmenes:
- `paddleocr_cache` → `/root/.paddleocr`
- `paddlex_cache` → `/root/.paddlex`

✅ Así no se descargan modelos cada vez.

---

## 6) Variables útiles ⚙️

En `docker-compose.yml` puedes ajustar:

- `GUNICORN_WORKERS` (recomendado 1 por OCR)
- `GUNICORN_THREADS` (3-8 según CPU)
- `GUNICORN_TIMEOUT` (120+ si OCR tarda)

---

## 7) Stop 🛑

```bash
docker compose down
```

Si quieres borrar también cache de modelos:

```bash
docker compose down -v
```
