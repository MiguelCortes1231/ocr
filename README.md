## ✨ Autor

👨‍💻 **Castillo Olivera Ricardo Orlando**  
🚀 Desarrollador de Software  
🇲🇽 México



# 🪪 INE / IFE OCR API 🇲🇽🤖

> **API profesional para extracción y validación de credenciales INE / IFE**  
> Construida en **Python + Flask + PaddleOCR**, pensada para **fotos reales desde celular**, con validaciones reales y lista para producción.

---

## 🚀 ¿Qué hace este proyecto?

✅ Extrae texto de imágenes del **ANVERSO** y **REVERSO** de la credencial INE/IFE  
✅ Detecta si **realmente es una INE válida**  
✅ Extrae campos clave como CURP, Clave de Elector, Vigencia, Sección, Domicilio  
✅ Soporta **MRZ (IDMEX...)** del reverso  
✅ Listo para usarse desde **React / Next / Vue**  
✅ Documentación automática con **Swagger**  
✅ Preparado para correr con **Gunicorn**

---

## 🧠 Arquitectura general

```
📦 ine-ocr-api
 ┣ 📜 main.py            # API Flask + OCR + validaciones
 ┣ 📜 requirements.txt   # Dependencias
 ┣ 📜 .gitignore         # Archivos ignorados
 ┣ 📜 README.md          # Este documento 😎
```

---

## 🧪 Tecnologías usadas

- 🐍 Python 3.10+
- 🌶️ Flask
- 🧾 Flasgger (Swagger UI)
- 🌐 flask-cors
- 👁️ PaddleOCR / PaddlePaddle
- 🧮 OpenCV
- 🦄 Gunicorn

---

## ⚙️ Requisitos previos

Antes de empezar asegúrate de tener:

- Python 3.10 o superior
- pip
- (Opcional) virtualenv

Verifica:
```bash
python --version
pip --version
```

---

## 🧪 Instalación paso a paso

### 1️⃣ Clona el repositorio

```bash
git clone https://github.com/tu-org/ine-ocr-api.git
cd ine-ocr-api
```

---

### 2️⃣ Crea entorno virtual 🧪

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
```

En Windows:
```bat
.venv\Scripts\activate
```

---

### 3️⃣ Instala dependencias 📦

```bash
pip install -r requirements.txt
```

⏳ *PaddleOCR puede tardar un poco, es normal*

---

## ▶️ Ejecutar en modo desarrollo

```bash
python main.py
```

📍 API disponible en:
```
http://localhost:5001
```

📘 Swagger:
```
http://localhost:5001/apidocs/
```

---

## 🚀 Ejecutar en producción con Gunicorn 🦄

> ⚠️ Recomendado para OCR: **1 worker + varios threads**

```bash
gunicorn -w 1 --threads 4 -b 0.0.0.0:5001 main:app
```

Versión extendida (timeout largo para OCR):

```bash
gunicorn \
  -w 1 --threads 4 \
  -b 0.0.0.0:5001 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  main:app
```

---

## 🌐 CORS (Frontend friendly)

El backend permite consumo desde cualquier frontend:

- React
- Next.js
- Vue
- Angular

Ejemplo desde React:

```ts
const formData = new FormData();
formData.append("imagen", file);

await axios.post("http://localhost:5001/ocr", formData);
```

---

## 🪪 Endpoints disponibles

### 📌 `POST /ocr` → ANVERSO

📸 Recibe imagen del frente de la INE

Campos devueltos:
- es_ine
- curp
- clave_elector
- fecha_nacimiento
- anio_registro
- seccion
- vigencia
- sexo
- pais
- calle
- numero
- colonia
- estado
- codigo_postal

---

### 📌 `POST /ocrreverso` → REVERSO

📸 Recibe imagen del reverso (MRZ)

Campos:
- es_ine
- linea1
- linea2
- apellido_paterno
- apellido_materno
- nombre_reverso

---

## 🧪 Ejemplo con curl

```bash
curl -X POST http://localhost:5001/ocr \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "imagen=@frente.jpg"
```

---

## 🧠 Notas importantes

⚠️ PaddleOCR descarga modelos automáticamente  
⚠️ No subas imágenes reales al repo  
⚠️ No subas modelos OCR a Git (ver .gitignore)

---

## 🔐 Seguridad

🔒 No expongas esta API públicamente sin:
- Rate limiting
- Autenticación
- HTTPS

---

## 🧹 Buenas prácticas

✔️ Usa `.env` para variables sensibles  
✔️ Mantén requirements.txt limpio  
✔️ Usa Gunicorn en producción  
✔️ Monitorea RAM (OCR consume memoria)

---

## 🏁 Roadmap futuro 🚧

- 🔍 Endpoint auto-detect frente/reverso
- 📊 Logs estructurados
- 🐳 Docker + docker-compose
- ☁️ Deploy en VPS / Cloud
- 🧠 Validaciones avanzadas de vigencia

---

## 👨‍💻 Autor

Desarrollado por **Ricardo Orlando Castillo Olivera**  
🇲🇽 México  
💻 Python · OCR · APIs · Automatización

---

## ⭐ Si este proyecto te sirve

Déjale una estrella ⭐  
y úsalo con responsabilidad 😉
