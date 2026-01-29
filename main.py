"""
🪪 INE/IFE OCR API (Flask + PaddleOCR) 🇲🇽
=================================================

✅ ¿Qué hace este API?
- Recibe una imagen (multipart/form-data) del **anverso** o **reverso** de una credencial INE.
- Usa PaddleOCR para extraer texto.
- Aplica heurísticas + regex para extraer campos relevantes.
- Expone Swagger UI con Flasgger para documentación y pruebas rápidas.

🚀 Endpoints
- POST /ocr         -> Procesa ANVERSO (con timeout + kill REAL del proceso)
- POST /ocrreverso  -> Procesa REVERSO (MRZ tipo "IDMEX...") (sin timeout por defecto)
- POST /enhance     -> Mejora imagen para OCR

🧠 Debug:
- POST /ocr?debug=1 -> incluye "_ocr_texts" (líneas crudas ya normalizadas)
- POST /ocrreverso?debug=1 -> idem

──────────────────────────────────────────────────────────────────────────────
🆕 MEJORA CLAVE solicitada ⏱️🪪
──────────────────────────────────────────────────────────────────────────────
🎯 Requisito:
- Si POST /ocr tarda más de X segundos:
  ✅ devolver {"error":"❌ La imagen es poco clara"} con HTTP 408
  ✅ y cortar de tajo el OCR (que NO se quede colgado)

✅ Implementación robusta:
- El OCR corre en un PROCESO separado (multiprocessing)
- Si se pasa del tiempo:
  🧨 se termina el proceso (terminate) y se responde 408
- Esto evita errores tipo:
  ❌ "cannot schedule new futures after shutdown"
  ❌ "se quedó colgado y el siguiente request ya no responde"

⚠️ Nota:
- Matar hilos de Python con ctypes es inseguro.
- Con procesos sí podemos matar el trabajo pesado de Paddle/OpenCV de forma confiable.

✨ Swagger:
- http://localhost:5001/apidocs/

"""

from __future__ import annotations

# ============================================================
# 🌐 Flask + Swagger + CORS
# ============================================================
from flask import Flask, request, jsonify, send_file
from flasgger import Swagger
from flask_cors import CORS

# ============================================================
# 🧠 OCR / Imagen
# ============================================================
from paddleocr import PaddleOCR
import numpy as np
import cv2

# ============================================================
# 🧩 Utils
# ============================================================
import re
import io
from typing import Dict, List, Optional, Any

# ============================================================
# 🧨 Timeout "kill real" con PROCESOS
# ============================================================
import multiprocessing as mp
import queue


# ============================================================
# ⚙️ Configuración Flask
# ============================================================
app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",  # ⚠️ en producción limita dominios
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "🪪 INE OCR API 🇲🇽",
        "description": "API para extraer datos del ANVERSO y REVERSO de credenciales INE/IFE usando PaddleOCR.",
        "version": "1.0.1",
    },
    "basePath": "/",
    "schemes": ["http"],
}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)


# ============================================================
# ⏱️ Timeout config
# ============================================================
OCR_TIMEOUT_SECONDS: int = 30


# ============================================================
# 🔎 OCR Engine (PaddleOCR)
# ============================================================
# ✅ Config que ya te funcionaba
# ⚠️ IMPORTANTE:
# - PaddleOCR dentro de procesos: se inicializa dentro del proceso worker (más estable)
# - Así evitamos compartir estado pesado entre requests.
def _build_ocr_engine() -> PaddleOCR:
    """
    🏭 Crea una instancia de PaddleOCR.
    Se llama dentro del proceso worker para evitar problemas de concurrencia.
    """
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="es",
    )


# ============================================================
# 🧩 Helpers de extracción (regex + utilidades)
# ============================================================
def buscar_en_lista(pattern: str, lista: List[str]) -> str:
    """🔍 Busca regex en una lista y regresa el primer group(1) encontrado."""
    for line in lista:
        match = re.search(pattern, line)
        if match:
            return match.group(1)
    return ""


def buscar_seccion(lista: List[str]) -> str:
    """🧾 Busca sección electoral (número de 4 dígitos)."""
    for line in lista:
        if re.fullmatch(r"\d{4}", line.strip()):
            return line.strip()
    return ""


def normalizar_textos(texts: List[str]) -> List[str]:
    """🧼 Normaliza líneas OCR (trim + espacios)."""
    limpios: List[str] = []
    for t in texts:
        t2 = re.sub(r"\s+", " ", (t or "").strip())
        if t2:
            limpios.append(t2)
    return limpios


# ============================================================
# 👤 Extracción robusta de NOMBRE (anverso)
# ============================================================
def _es_linea_candidata_nombre(line: str) -> bool:
    """
    🧪 Heurística: determina si una línea parece nombre.
    """
    s = (line or "").strip()
    if not s:
        return False

    up = s.upper()

    if re.search(r"\d", up):
        return False

    if re.search(
        r"(INSTITUTO|NACIONAL|ELECTORAL|CREDENCIAL|VOTAR|MÉXICO|MEXICO|DOMICILIO|CLAVE|ELECTOR|CURP|SEXO|FECHA|NACIMIENTO|REGISTRO|SECCI[ÓO]N|VIGENCIA|AÑO)",
        up
    ):
        return False

    if not re.fullmatch(r"[A-ZÁÉÍÓÚÜÑ\s\.\-]+", up):
        return False

    if len(up) < 3:
        return False

    return True


def _limpiar_nombre_pieza(s: str) -> str:
    """🧽 Limpia pieza de nombre."""
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\:\-\.\,]+", "", s).strip()
    s = re.sub(r"[\:\-\.\,]+$", "", s).strip()
    return s


def extraer_nombre_completo(texts: List[str]) -> str:
    """
    👤 Extrae el nombre completo del ANVERSO.
    """
    if not texts:
        return ""

    lines = normalizar_textos(texts)
    upper = [l.upper() for l in lines]

    patron_nombre_label = r"\bN[O0]M[B8]R[E3](?:\(?S\)?)?\b"
    patron_domicilio_label = r"\bDOMICILIO\b"

    idx_nombre: Optional[int] = None
    idx_domicilio: Optional[int] = None

    for i, l in enumerate(upper):
        if idx_nombre is None and re.search(patron_nombre_label, l):
            idx_nombre = i
        if idx_domicilio is None and re.search(patron_domicilio_label, l):
            idx_domicilio = i

    def juntar(cands: List[str]) -> str:
        cands = [_limpiar_nombre_pieza(c) for c in cands if _limpiar_nombre_pieza(c)]
        cands = [c for c in cands if _es_linea_candidata_nombre(c)]

        dedup: List[str] = []
        for c in cands:
            if not dedup or dedup[-1].upper() != c.upper():
                dedup.append(c)

        nombre = re.sub(r"\s+", " ", " ".join(dedup)).strip()

        # 🧼 quitar label NOMBRE si quedó pegado
        nombre = re.sub(
            r"^\s*N[O0]M[B8]R[E3](?:\(?S\)?)?\s*[:\-]?\s*",
            "",
            nombre,
            flags=re.IGNORECASE
        ).strip()

        return nombre

    if idx_nombre is not None:
        candidatos: List[str] = []

        same_line = lines[idx_nombre]
        same_up = upper[idx_nombre]
        m = re.search(patron_nombre_label + r"[:\s\-]*", same_up)
        if m:
            resto = _limpiar_nombre_pieza(same_line[m.end():])
            if resto and _es_linea_candidata_nombre(resto):
                candidatos.append(resto)

        start = idx_nombre + 1
        end = idx_domicilio if (idx_domicilio is not None and idx_domicilio > start) else len(lines)

        for j in range(start, min(end, start + 6)):
            if re.search(r"\b(DOMICILIO|CLAVE|ELECTOR|CURP|SEXO|FECHA|NACIMIENTO|REGISTRO|SECCI[ÓO]N|VIGENCIA|AÑO)\b", upper[j]):
                break

            pieza = _limpiar_nombre_pieza(lines[j])
            if pieza and _es_linea_candidata_nombre(pieza):
                candidatos.append(pieza)

            if len(candidatos) >= 4:
                break

        nombre = juntar(candidatos)

        if nombre:
            piezas = nombre.split()
            if len(candidatos) <= 1 or len(piezas) <= 2:
                if idx_domicilio is not None:
                    extra: List[str] = []
                    for k in range(idx_domicilio - 1, idx_nombre, -1):
                        pieza2 = _limpiar_nombre_pieza(lines[k])
                        if pieza2 and _es_linea_candidata_nombre(pieza2):
                            extra.append(pieza2)
                        if len(extra) >= 3:
                            break
                    extra = list(reversed(extra))
                    if extra:
                        nombre2 = juntar(extra + candidatos)
                        if nombre2:
                            return nombre2

            return nombre

    if idx_domicilio is not None:
        candidatos = []
        for j in range(idx_domicilio - 1, max(idx_domicilio - 7, -1), -1):
            if j < 0:
                break
            pieza = _limpiar_nombre_pieza(lines[j])
            if pieza and _es_linea_candidata_nombre(pieza):
                candidatos.append(pieza)
            if len(candidatos) >= 4:
                break
        candidatos = list(reversed(candidatos))
        return juntar(candidatos)

    candidatos = []
    for l in lines:
        pieza = _limpiar_nombre_pieza(l)
        if pieza and _es_linea_candidata_nombre(pieza):
            candidatos.append(pieza)
        if len(candidatos) >= 3:
            break

    return juntar(candidatos)


# ============================================================
# 🪪 Extracción ANVERSO
# ============================================================
def extraer_campos_ine(texts: List[str]) -> Dict[str, Any]:
    """🪪 Extrae campos típicos del ANVERSO de la credencial INE."""
    texts = normalizar_textos(texts)
    es_ine = any("INSTITUTO NACIONAL ELECTORAL" in line.upper() for line in texts)

    nombre_completo = extraer_nombre_completo(texts)

    campos: Dict[str, Any] = {
        "es_ine": es_ine,
        "nombre": nombre_completo,
        "curp": buscar_en_lista(r"([A-Z]{4}[0-9]{6}[HMX]{1}[A-Z]+[0-9]+)", texts),
        "clave_elector": buscar_en_lista(r"\b([A-Z]{6}\d{6,8}[A-Z0-9]{2,4})\b", texts),
        "fecha_nacimiento": buscar_en_lista(r"\b(\d{2}/\d{2}/\d{4})\b", texts),
        "anio_registro": buscar_en_lista(r"(\d{4}\s\d+)", texts),
        "seccion": buscar_seccion(texts),
        "vigencia": buscar_en_lista(r"(\d{4}\s[-]?\s?\d{4})", texts),
        "sexo": buscar_en_lista(r"\b(H|M|X)\b", texts),
        "pais": "Mex",
    }

    dom_index: Optional[int] = next(
        (i for i, line in enumerate(texts) if "DOMICILIO" in line.upper()),
        None,
    )

    if dom_index is not None:
        campos["calle"] = texts[dom_index + 1] if len(texts) > dom_index + 1 else ""
        campos["colonia"] = texts[dom_index + 2] if len(texts) > dom_index + 2 else ""
        campos["estado"] = texts[dom_index + 3] if len(texts) > dom_index + 3 else ""
    else:
        campos["calle"] = ""
        campos["colonia"] = ""
        campos["estado"] = ""

    match_num = re.search(r"\b(\d{1,5}[A-Z]?(?:\s*INT\.?\s*\d+)?)\b", campos["calle"])
    campos["numero"] = match_num.group(1) if match_num else ""

    campos["codigo_postal"] = buscar_en_lista(r"\b(\d{5})\b", [campos["colonia"], campos["estado"]])

    return campos


# ============================================================
# 🔙 Extracción REVERSO (MRZ)
# ============================================================
def extraer_campos_reverso(texto: List[str]) -> Dict[str, Any]:
    """🔙 Extrae información del REVERSO mediante MRZ tipo IDMEX..."""
    texto = normalizar_textos(texto)

    resultado: Dict[str, Any] = {
        "linea1": "",
        "linea2": "",
        "apellido_paterno": "",
        "apellido_materno": "",
        "nombre_reverso": "",
        "es_ine": False,
    }

    if len(texto) != 3 or not texto[0].startswith("IDMEX"):
        return resultado

    resultado["es_ine"] = True
    resultado["linea1"] = texto[0]
    resultado["linea2"] = texto[1]

    line3 = texto[2]
    name_parts = [p for p in line3.split("<") if p]

    if len(name_parts) >= 1:
        resultado["apellido_paterno"] = name_parts[0]
    if len(name_parts) >= 2:
        resultado["apellido_materno"] = name_parts[1]
    if len(name_parts) >= 3:
        resultado["nombre_reverso"] = "".join(name_parts[2:])

    return resultado


# ============================================================
# 🖼️ Lectura de imagen desde request
# ============================================================
def leer_imagen_desde_request(field_name: str = "imagen") -> Optional[np.ndarray]:
    """🖼️ Lee imagen multipart/form-data y decodifica con OpenCV."""
    if field_name not in request.files:
        return None

    file = request.files[field_name]
    data = file.read()
    if not data:
        return None

    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img


# ============================================================
# 🖼️ Mejora de imagen (igual que tu pipeline)
# ============================================================
def _order_points(pts: np.ndarray) -> np.ndarray:
    """🧭 Ordena 4 puntos: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """📐 Warp de perspectiva usando 4 puntos."""
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def auto_recortar_ine(img_bgr: np.ndarray) -> np.ndarray:
    """✂️ Detecta contorno de credencial y corrige perspectiva; si falla regresa original."""
    original = img_bgr
    h, w = original.shape[:2]
    max_side = max(h, w)
    scale = 1100 / max_side if max_side > 1100 else 1.0
    img = cv2.resize(original, (int(w * scale), int(h * scale))) if scale != 1.0 else original.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 160)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return original

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]

    screen_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is None:
        return original

    pts = screen_cnt.reshape(4, 2).astype("float32")
    if scale != 1.0:
        pts = pts / scale

    warped = _four_point_transform(original, pts)

    wh, ww = warped.shape[:2]
    pad = int(min(wh, ww) * 0.01)
    if pad > 0 and wh > 2 * pad and ww > 2 * pad:
        warped = warped[pad:wh - pad, pad:ww - pad]

    return warped


def mejorar_para_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """🧠 Mejora suave para OCR (denoise + contraste + nitidez)."""
    img = img_bgr.copy()

    h, w = img.shape[:2]
    target_w = 1400
    if w < target_w:
        scale = target_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    img = cv2.addWeighted(img, 1.55, blur, -0.55, 0)

    return img


def pipeline_mejora_ine(img_bgr: np.ndarray) -> np.ndarray:
    """🧪 Pipeline final: recorte + mejora."""
    return mejorar_para_ocr(auto_recortar_ine(img_bgr))


# ============================================================
# 🧨 Worker OCR en PROCESO (para poder MATARLO)
# ============================================================
def _ocr_worker(img_bgr: np.ndarray, out_q: mp.Queue) -> None:
    """
    🏗️ Proceso worker:
    - Inicializa PaddleOCR adentro del proceso
    - Ejecuta predict
    - Regresa textos por Queue
    """
    try:
        engine = _build_ocr_engine()
        result = engine.predict(img_bgr)
        texts = result[0]["rec_texts"] if result else []
        out_q.put({"ok": True, "texts": texts})
    except Exception as e:
        out_q.put({"ok": False, "error": str(e)})


def predict_ocr_texts_with_timeout_kill(img_bgr: np.ndarray, timeout_seconds: int) -> List[str]:
    """
    ⏱️ Ejecuta OCR en proceso y lo MATA si se pasa del tiempo.
    """
    out_q: mp.Queue = mp.Queue(maxsize=1)
    p = mp.Process(target=_ocr_worker, args=(img_bgr, out_q), daemon=True)

    p.start()
    p.join(timeout_seconds)

    if p.is_alive():
        # 🧨 TIMEOUT: matar proceso
        try:
            p.terminate()
        finally:
            p.join(timeout=2)
        raise TimeoutError("OCR tardó demasiado (proceso terminado)")

    # Proceso terminó: leer resultado
    try:
        payload = out_q.get_nowait()
    except queue.Empty:
        raise RuntimeError("OCR terminó pero no devolvió resultado")

    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "Error desconocido en OCR"))

    return payload.get("texts") or []


# ============================================================
# 🖼️ Endpoint: Enhance
# ============================================================
@app.route("/enhance", methods=["POST"])
def enhance_image():
    """
    🖼️ Mejora imagen para OCR (INE/IFE)
    ---
    tags:
      - Image Enhance
    consumes:
      - multipart/form-data
    parameters:
      - name: imagen
        in: formData
        type: file
        required: true
        description: 📸 Foto original (celular) de la credencial
    responses:
      200:
        description: ✅ Imagen mejorada (PNG) lista para /ocr
      400:
        description: ❌ Error de entrada
    """
    img = leer_imagen_desde_request("imagen")
    if img is None:
        return jsonify({"error": "❌ No se envió la imagen o está vacía"}), 400

    enhanced = pipeline_mejora_ine(img)

    ok, buf = cv2.imencode(".png", enhanced)
    if not ok:
        return jsonify({"error": "❌ No se pudo codificar la imagen mejorada"}), 400

    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/png",
        as_attachment=False,
        download_name="ine_enhanced.png",
    )


# ============================================================
# 🚀 Endpoint: OCR ANVERSO (con timeout + kill)
# ============================================================
@app.route("/ocr", methods=["POST"])
def ocr_anverso():
    """
    🪪 OCR ANVERSO (INE) ⏱️🧨
    ---
    tags:
      - INE OCR
    consumes:
      - multipart/form-data
    parameters:
      - name: imagen
        in: formData
        type: file
        required: true
        description: 📸 Imagen del anverso de la credencial
    responses:
      200:
        description: ✅ Datos extraídos del anverso
      400:
        description: ❌ Falta imagen o imagen inválida
      408:
        description: ⏱️ OCR tardó demasiado -> la imagen es poco clara
    """
    img = leer_imagen_desde_request("imagen")
    if img is None:
        return jsonify({"error": "❌ No se envió la imagen o está vacía"}), 400

    try:
        texts = predict_ocr_texts_with_timeout_kill(img, OCR_TIMEOUT_SECONDS)
    except TimeoutError:
        return jsonify({"error": "❌ La imagen es poco clara"}), 408
    except Exception as e:
        return jsonify({"error": f"❌ Error procesando OCR: {str(e)}"}), 400

    datos = extraer_campos_ine(texts)

    if (request.args.get("debug") or "").strip() in ("1", "true", "True", "yes", "YES"):
        datos["_ocr_texts"] = normalizar_textos(texts)

    return jsonify(datos)


# ============================================================
# 🚀 Endpoint: OCR REVERSO (sin timeout por defecto)
# ============================================================
@app.route("/ocrreverso", methods=["POST"])
def ocr_reverso():
    """
    🔙 OCR REVERSO (INE - MRZ)
    ---
    tags:
      - INE OCR
    consumes:
      - multipart/form-data
    parameters:
      - name: imagen
        in: formData
        type: file
        required: true
        description: 📸 Imagen del reverso de la credencial (MRZ "IDMEX...")
    responses:
      200:
        description: ✅ Datos extraídos del reverso (MRZ)
      400:
        description: ❌ Falta imagen o imagen inválida
    """
    img = leer_imagen_desde_request("imagen")
    if img is None:
        return jsonify({"error": "❌ No se envió la imagen o está vacía"}), 400

    try:
        # Aquí usamos OCR sin timeout (pero en el MISMO proceso).
        # Si quieres también con kill, te lo adapto igual.
        engine = _build_ocr_engine()
        result = engine.predict(img)
        texts = result[0]["rec_texts"] if result else []
    except Exception as e:
        return jsonify({"error": f"❌ Error procesando OCR: {str(e)}"}), 400

    datos = extraer_campos_reverso(texts)

    if (request.args.get("debug") or "").strip() in ("1", "true", "True", "yes", "YES"):
        datos["_ocr_texts"] = normalizar_textos(texts)

    return jsonify(datos)


# ============================================================
# 🩺 Health
# ============================================================
@app.route("/health", methods=["GET"])
def health_check():
    """
    🩺 Health Check del API
    ---
    tags:
      - System
    responses:
      200:
        description: ✅ API funcionando correctamente
    """
    return jsonify({"status": "✅ OK", "service": "INE OCR API", "version": "1.0.1"})


# ============================================================
# ▶️ Run
# ============================================================
if __name__ == "__main__":
    # 🧠 Recomendación:
    # - debug=False en producción
    # - Si usas gunicorn, configura workers apropiados
    app.run(host="0.0.0.0", port=5001, debug=False)
