"""
🪪 INE/IFE OCR API MEJORADO 🇲🇽
=================================================

✅ MEJORAS IMPLEMENTADAS:
1. Clasificación automática de tipo de credencial (C, D, GM)
2. Validación y completado de datos desde CURP y Clave de Elector
3. Mejora en extracción de nombre (filtra palabras erróneas)
4. Reglas específicas por tipo de credencial
5. Mayor precisión en extracción de campos

🚀 Endpoints:
- POST /ocr  -> Procesa ANVERSO con todas las mejoras
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
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

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
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "🪪 INE OCR API MEJORADO 🇲🇽",
        "description": "API mejorada para extraer datos de credenciales INE/IFE con validación desde CURP y Clave de Elector",
        "version": "2.0.0",
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
# 📊 DICCIONARIOS DE REFERENCIA
# ============================================================
CODIGOS_ESTADO_CURP = {
    'AS': 'AGUASCALIENTES', 'BC': 'BAJA CALIFORNIA', 'BS': 'BAJA CALIFORNIA SUR',
    'CC': 'CAMPECHE', 'CL': 'COAHUILA', 'CM': 'COLIMA', 'CS': 'CHIAPAS',
    'CH': 'CHIHUAHUA', 'DF': 'CIUDAD DE MÉXICO', 'DG': 'DURANGO',
    'GT': 'GUANAJUATO', 'GR': 'GUERRERO', 'HG': 'HIDALGO', 'JC': 'JALISCO',
    'MC': 'MÉXICO', 'MN': 'MICHOACÁN', 'MS': 'MORELOS', 'NT': 'NAYARIT',
    'NL': 'NUEVO LEÓN', 'OC': 'OAXACA', 'PL': 'PUEBLA', 'QT': 'QUERÉTARO',
    'QR': 'QUINTANA ROO', 'SP': 'SAN LUIS POTOSÍ', 'SL': 'SINALOA',
    'SR': 'SONORA', 'TC': 'TABASCO', 'TS': 'TAMAULIPAS', 'TL': 'TLAXCALA',
    'VZ': 'VERACRUZ', 'YN': 'YUCATÁN', 'ZS': 'ZACATECAS', 'NE': 'EXTRANJERO'
}

CODIGOS_ESTADO_ELECTOR = {
    '01': 'AGUASCALIENTES', '02': 'BAJA CALIFORNIA', '03': 'BAJA CALIFORNIA SUR',
    '04': 'CAMPECHE', '05': 'COAHUILA', '06': 'COLIMA', '07': 'CHIAPAS',
    '08': 'CHIHUAHUA', '09': 'CIUDAD DE MÉXICO', '10': 'DURANGO',
    '11': 'GUANAJUATO', '12': 'GUERRERO', '13': 'HIDALGO', '14': 'JALISCO',
    '15': 'MÉXICO', '16': 'MICHOACÁN', '17': 'MORELOS', '18': 'NAYARIT',
    '19': 'NUEVO LEÓN', '20': 'OAXACA', '21': 'PUEBLA', '22': 'QUERÉTARO',
    '23': 'QUINTANA ROO', '24': 'SAN LUIS POTOSÍ', '25': 'SINALOA',
    '26': 'SONORA', '27': 'TABASCO', '28': 'TAMAULIPAS', '29': 'TLAXCALA',
    '30': 'VERACRUZ', '31': 'YUCATÁN', '32': 'ZACATECAS'
}

# ============================================================
# 🔍 OCR Engine (PaddleOCR)
# ============================================================
def _build_ocr_engine() -> PaddleOCR:
    """🏭 Crea una instancia de PaddleOCR."""
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="es",
    )


# ============================================================
# 🏷️ CLASIFICACIÓN DE TIPO DE CREDENCIAL
# ============================================================
def clasificar_tipo_credencial(texts: List[str]) -> str:
    """
    🔍 Clasifica la credencial en tipo C, D o GM basado en patrones.
    
    Reglas:
    - GM: Tiene "INSTITUTO NACIONAL ELECTORAL" y estructura específica
    - D: Tiene "INSTITUTO NACIONAL ELECTORAL" pero formato diferente
    - C: Tiene "INSTITUTO FEDERAL ELECTORAL" (más antiguo)
    """
    textos_upper = [t.upper() for t in texts]
    texto_completo = " ".join(textos_upper)
    
    # Patrones para identificar tipo
    tiene_ine = "INSTITUTO NACIONAL ELECTORAL" in texto_completo
    tiene_ife = "INSTITUTO FEDERAL ELECTORAL" in texto_completo
    tiene_credencial_para_votar = "CREDENCIAL PARA VOTAR" in texto_completo
    tiene_mrz_idmex = "IDMEX" in texto_completo
    
    # Heurísticas para diferenciar GM vs D
    if tiene_ine and tiene_credencial_para_votar:
        # GM suele tener "CLAVE DE ELECTOR" y estructura más organizada
        if "CLAVE DE ELECTOR" in texto_completo:
            return "GM"
        else:
            return "D"
    elif tiene_ife:
        return "C"
    
    # Por defecto, si no se identifica claramente
    return "D"


# ============================================================
# 🧠 VALIDACIÓN Y EXTRACCIÓN DESDE CURP
# ============================================================
def extraer_datos_desde_curp(curp: str) -> Dict[str, str]:
    """
    📊 Extrae información validada desde la CURP.
    
    Estructura CURP: AAAA BB CC DD E F G H I J K L M N Ñ O P
    """
    datos = {
        "sexo": "",
        "fecha_nacimiento": "",
        "entidad_nacimiento": "",
        "estado": ""
    }
    
    if not curp or len(curp) < 16:
        return datos
    
    # 1. Sexo (10º carácter)
    if len(curp) >= 10:
        sexo_char = curp[10].upper()
        if sexo_char == 'H':
            datos["sexo"] = "H"
        elif sexo_char == 'M':
            datos["sexo"] = "M"
        else:
            datos["sexo"] = "X"
    
    # 2. Fecha de nacimiento (5º al 10º carácter: AAMMDD)
    if len(curp) >= 10:
        anio = curp[4:6]  # Últimos 2 dígitos del año
        mes = curp[6:8]
        dia = curp[8:10]
        
        # Determinar siglo (19xx o 20xx)
        # Asumimos que si el año es mayor a año actual - 100, es 1900, sino 2000
        año_actual_2dig = datetime.now().year % 100
        año_num = int(anio)
        siglo = "19" if año_num > año_actual_2dig else "20"
        
        datos["fecha_nacimiento"] = f"{dia}/{mes}/{siglo}{anio}"
    
    # 3. Entidad de nacimiento (12º y 13º carácter)
    if len(curp) >= 13:
        codigo_estado = curp[11:13].upper()
        datos["entidad_nacimiento"] = codigo_estado
        datos["estado"] = CODIGOS_ESTADO_CURP.get(codigo_estado, "")
    
    return datos


# ============================================================
# 🗳️ VALIDACIÓN Y EXTRACCIÓN DESDE CLAVE DE ELECTOR
# ============================================================
def extraer_datos_desde_clave_elector(clave: str) -> Dict[str, str]:
    """
    📍 Extrae información desde la Clave de Elector.
    
    Estructura: AAAA BB CCC DD E F
    """
    datos = {
        "estado_clave": "",
        "seccion_clave": "",
        "anio_registro_clave": ""
    }
    
    if not clave or len(clave) < 13:
        return datos
    
    # 1. Estado (primeros 2 dígitos)
    if len(clave) >= 2:
        codigo_estado = clave[0:2]
        datos["estado_clave"] = CODIGOS_ESTADO_ELECTOR.get(codigo_estado, "")
    
    # 2. Sección (posiciones 5-6, considerando variaciones)
    # Buscar 4 dígitos consecutivos que puedan ser sección
    seccion_match = re.search(r'\b(\d{4})\b', clave)
    if seccion_match:
        datos["seccion_clave"] = seccion_match.group(1)
    
    # 3. Año de registro (varía según posición)
    # Buscar patrón de 4 dígitos que sea un año plausible (1900-2025)
    for match in re.finditer(r'\b(19\d{2}|20[0-2]\d)\b', clave):
        año = int(match.group())
        if 1900 <= año <= datetime.now().year + 1:
            datos["anio_registro_clave"] = str(año)
            break
    
    return datos


# ============================================================
# 👤 MEJORA EN EXTRACCIÓN DE NOMBRE
# ============================================================
def limpiar_y_validar_nombre(nombre: str) -> str:
    """
    🧹 Limpia y valida el nombre extraído, removiendo palabras erróneas.
    """
    if not nombre:
        return ""
    
    # Palabras que NO deberían estar en un nombre

    
    palabras_invalidas = [
    'EDAD', 'AÑOS', 'AÑO', 'EDAD:', 'EDADES', 'FECHA', 'NACIMIENTO',
    'DOMICILIO', 'CALLE', 'COLONIA', 'ESTADO', 'MUNICIPIO', 'CIUDAD',
    'CP', 'C.P.', 'CÓDIGO', 'POSTAL', 'SECCIÓN', 'SECCION', 'CLAVE',
    'ELECTOR', 'CURP', 'VIGENCIA', 'VIGENTE', 'INSTITUTO', 'NACIONAL',
    'FEDERAL', 'ELECTORAL', 'CREDENCIAL', 'VOTAR', 'PARA', 'MÉXICO',
    'REGISTRO'  # ✅ evita "DE REGISTRO"
    ]
    
    # Convertir a mayúsculas para comparación
    nombre_upper = nombre.upper()
    
    # Remover palabras inválidas
    palabras = nombre_upper.split()
    palabras_limpias = []
    
    for palabra in palabras:
        palabra_limpia = re.sub(r'[^\wÁÉÍÓÚÜÑ]', '', palabra)
        if (palabra_limpia and 
            len(palabra_limpia) > 1 and 
            palabra_limpia not in palabras_invalidas and
            not palabra_limpia.isdigit() and
            not re.match(r'^\d+[A-Z]*$', palabra_limpia)):
            palabras_limpias.append(palabra)
    
    # Reconstruir nombre manteniendo capitalización original
    nombre_original = nombre.split()
    nombre_final = []
    
    for palabra in nombre_original:
        if palabra.upper() in [p.upper() for p in palabras_limpias]:
            nombre_final.append(palabra)
    
    return " ".join(nombre_final)


# ============================================================
# 👤 CORRECCIÓN: EXTRACCIÓN DE NOMBRE PARA TIPO GM
# ============================================================
def extraer_nombre_mejorado(texts: List[str], tipo_credencial: str) -> str:
    """
    👤 Extrae y limpia el nombre según el tipo de credencial.

    ✅ FIX GM robusto:
    - Si OCR NO detecta bien "NOMBRE", usamos un ancla: "DOMICILIO"
      y tomamos las líneas inmediatamente anteriores como nombre (2-4 líneas).
    - Evita devolver encabezados tipo: "INSTITUTO NACIONAL ELECTORAL".
    """
    textos_limpios = normalizar_textos(texts)

    # ============================================================
    # 🪪 ESTRATEGIA GM (prioritaria)
    # ============================================================
    if tipo_credencial == "GM":

        # 🚫 Frases/etiquetas que NO son nombre
        blacklist_regex = r'(INSTITUTO|NACIONAL|ELECTORAL|CREDENCIAL|PARA\s+VOTAR|M[EÉ]XICO|ESTADOS\s+UNIDOS)'
        stop_labels_regex = r'(DOMICILIO|CLAVE|CURP|FECHA|SECCI[ÓO]N|AÑO|VIGENCIA|SEXO)'

        # ------------------------------------------------------------
        # ✅ ESTRATEGIA 0 (NUEVA): ANCLA POR "DOMICILIO"
        # Toma 2–4 líneas ANTES de "DOMICILIO" como nombre
        # ------------------------------------------------------------
        idx_dom = None
        for i, line in enumerate(textos_limpios):
            if "DOMICILIO" in line.upper():
                idx_dom = i
                break

        if idx_dom is not None:
            # Revisar hasta 8 líneas antes de DOMICILIO
            ventana = textos_limpios[max(0, idx_dom - 10):idx_dom]

            # Filtrar basura/encabezados
            candidatos = []
            for s in ventana:
                s_clean = s.strip()
                s_up = s_clean.upper()

                if not s_clean:
                    continue

                # saltar etiquetas
                if re.search(stop_labels_regex, s_up):
                    continue

                # saltar encabezados institucionales
                if re.search(blacklist_regex, s_up):
                    continue

                # saltar si tiene números
                if any(ch.isdigit() for ch in s_up):
                    continue

                # saltar si es demasiado corto (ruido)
                if len(re.sub(r'[^A-ZÁÉÍÓÚÜÑ]', '', s_up)) < 2:
                    continue

                # saltar si literal dice NOMBRE
                if re.fullmatch(r'NOMBRE', s_up):
                    continue

                candidatos.append(s_clean)

            # Queremos las ÚLTIMAS 2-4 líneas antes de DOMICILIO (ahí suele estar el nombre)
            if candidatos:
                partes = candidatos[-4:]  # máximo 4 líneas
                nombre_candidato = " ".join(partes).strip()

                # Validación mínima: 2+ palabras
                if len(nombre_candidato.split()) >= 2:
                    return nombre_candidato

        # ------------------------------------------------------------
        # ✅ Caso A: "NOMBRE" en línea sola y el nombre viene abajo en varias líneas
        # ------------------------------------------------------------
        for i, line in enumerate(textos_limpios):
            line_upper = line.upper().strip()

            if re.fullmatch(r'^NOMBRE\s*$', line_upper):
                partes: List[str] = []

                for j in range(i + 1, min(i + 6, len(textos_limpios))):
                    s = textos_limpios[j].strip()
                    s_up = s.upper().strip()

                    if re.search(stop_labels_regex, s_up):
                        break

                    if not s:
                        continue

                    if re.search(blacklist_regex, s_up):
                        continue

                    if any(ch.isdigit() for ch in s_up):
                        continue

                    if len(re.sub(r'[^A-ZÁÉÍÓÚÜÑ]', '', s_up)) < 2:
                        continue

                    partes.append(s)

                nombre_candidato = " ".join(partes).strip()
                if len(nombre_candidato.split()) >= 2:
                    return nombre_candidato

        # ------------------------------------------------------------
        # ✅ Caso B: "NOMBRE: JUAN PEREZ ..." en misma línea
        # ------------------------------------------------------------
        for line in textos_limpios:
            line_upper = line.upper()
            match = re.search(r'NOMBRE\s*[:\-]?\s*([A-ZÁÉÍÓÚÜÑ\s\.]{3,})', line_upper)
            if match:
                nombre_candidato = match.group(1).strip()

                if (nombre_candidato and
                    not re.search(stop_labels_regex, nombre_candidato.upper()) and
                    not re.search(blacklist_regex, nombre_candidato.upper()) and
                    not any(ch.isdigit() for ch in nombre_candidato) and
                    len(nombre_candidato.split()) >= 2):
                    return nombre_candidato

        # Si GM falla, seguimos con fallback general

    # ============================================================
    # 🧠 ESTRATEGIA GENERAL (C/D o fallback)
    # ============================================================
    patrones_nombre = [
        r'NOMBRE[:\s\-]*([A-ZÁÉÍÓÚÜÑ\s\.]{5,})',
        r'^([A-ZÁÉÍÓÚÜÑ]{2,}\s+[A-ZÁÉÍÓÚÜÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÜÑ]{2,}){0,3})$'
    ]

    for patron in patrones_nombre:
        for line in textos_limpios:
            up = line.upper().strip()

            # evitar encabezados institucionales
            if re.search(r'(INSTITUTO|NACIONAL|ELECTORAL|CREDENCIAL|PARA\s+VOTAR|M[EÉ]XICO)', up):
                continue

            match = re.search(patron, up)
            if match:
                nombre = match.group(1).strip() if match.groups() else match.group(0).strip()

                if (nombre and
                    len(nombre.split()) >= 2 and
                    not re.search(r'(DOMICILIO|CLAVE|CURP|FECHA|SECCI[ÓO]N|AÑO|REGISTRO|VIGENCIA|SEXO)', nombre.upper()) and
                    not re.search(r'(INSTITUTO|NACIONAL|ELECTORAL|CREDENCIAL|PARA\s+VOTAR|M[EÉ]XICO)', nombre.upper()) and
                    not any(ch.isdigit() for ch in nombre)):
                    return nombre

    # ============================================================
    # 🧨 FALLBACK FINAL (último recurso)
    # ============================================================
    candidatos = []
    for line in textos_limpios:
        up = line.upper().strip()
        if not up:
            continue
        if len(up.split()) < 2:
            continue
        if re.search(r'(DOMICILIO|CLAVE|CURP|FECHA|SECCI[ÓO]N|AÑO|REGISTRO|VIGENCIA|SEXO)', up):
            continue
        if re.search(r'(INSTITUTO|NACIONAL|ELECTORAL|CREDENCIAL|PARA\s+VOTAR|M[EÉ]XICO)', up):
            continue
        if any(ch.isdigit() for ch in up):
            continue
        candidatos.append(line.strip())

    if candidatos:
        return candidatos[0]

    return ""

# ============================================================
# 📅 CORRECCIÓN: EXTRACCIÓN DE VIGENCIA
# ============================================================
def extraer_vigencia_correcta(texts: List[str], tipo_credencial: str) -> str:
    """
    📅 Extrae correctamente la vigencia de la credencial.
    CORREGIDO: Maneja específicamente formato "2021 - 2031"
    """
    textos_limpios = normalizar_textos(texts)
    
    # Buscar patrón específico de vigencia
    for line in textos_limpios:
        line_upper = line.upper()
        
        # Buscar línea que contenga "VIGENCIA"
        if "VIGENCIA" in line_upper:
            # Intentar extraer de la misma línea
            match = re.search(r'VIGENCIA\s*[:\-]?\s*(\d{4}\s*[-\s]+\s*\d{4})', line_upper)
            if match:
                vigencia = match.group(1)
                # Limpiar formato
                vigencia = re.sub(r'\s+', ' ', vigencia.replace('-', ' - ').strip())
                return vigencia
            
            # Si no está en la misma línea, buscar en siguientes líneas
            idx = textos_limpios.index(line)
            for j in range(idx + 1, min(idx + 3, len(textos_limpios))):
                siguiente = textos_limpios[j]
                # Buscar patrón de dos años separados por guión
                match = re.search(r'(\d{4}\s*[-\s]+\s*\d{4})', siguiente)
                if match:
                    vigencia = match.group(1)
                    vigencia = re.sub(r'\s+', ' ', vigencia.replace('-', ' - ').strip())
                    return vigencia
        
        # Buscar directamente patrón de años con guión
        match = re.search(r'\b(\d{4}\s*[-]\s*\d{4})\b', line)
        if match:
            # Verificar que sean años plausibles (1900-2099)
            años = re.findall(r'\d{4}', match.group(1))
            if len(años) == 2:
                año1, año2 = int(años[0]), int(años[1])
                if 1900 <= año1 <= 2099 and 1900 <= año2 <= 2099 and año2 > año1:
                    vigencia = match.group(1)
                    vigencia = re.sub(r'\s+', ' ', vigencia.replace('-', ' - ').strip())
                    return vigencia
    
    # Buscar patrón "VIGENCIA" seguido de años
    for i, line in enumerate(textos_limpios):
        if "VIGENCIA" in line.upper():
            # Revisar próximas 3 líneas
            for j in range(i, min(i + 3, len(textos_limpios))):
                siguiente = textos_limpios[j]
                # Buscar cualquier patrón de año
                años = re.findall(r'\b(19\d{2}|20\d{2})\b', siguiente)
                if len(años) >= 2:
                    return f"{años[0]} - {años[1]}"
                elif len(años) == 1 and j > i:
                    # Si solo hay un año en línea siguiente, podría ser inicio de vigencia
                    siguiente2 = textos_limpios[j + 1] if j + 1 < len(textos_limpios) else ""
                    año2_match = re.search(r'\b(19\d{2}|20\d{2})\b', siguiente2)
                    if año2_match:
                        return f"{años[0]} - {año2_match.group(1)}"
    
    return ""


# ============================================================
# 🪪 FUNCIÓN PRINCIPAL CORREGIDA
# ============================================================
def extraer_campos_ine_mejorado(texts: List[str]) -> Dict[str, Any]:
    """
    🪪 Extrae campos del ANVERSO con validación desde CURP y Clave de Elector.
    CORREGIDO: Nombre y vigencia.
    """
    # Normalizar textos una sola vez
    textos_limpios = normalizar_textos(texts)
    
    # 1. Clasificar tipo de credencial
    tipo_credencial = clasificar_tipo_credencial(textos_limpios)
    
    # 2. Extraer CURP y Clave de Elector (usar textos_limpios)
    curp_crudo = buscar_en_lista(r'([A-Z]{4}[0-9]{6}[HMX][A-Z]{5,6}[0-9A-Z])', textos_limpios)
    clave_elector_crudo = buscar_en_lista(r'\b([A-Z0-9]{18})\b', textos_limpios) or buscar_en_lista(r'\b([A-Z]{6}\d{8,10}[A-Z0-9]{2,4})\b', textos_limpios)
    
    # 3. Extraer datos desde CURP y Clave de Elector
    datos_curp = extraer_datos_desde_curp(curp_crudo)
    datos_clave = extraer_datos_desde_clave_elector(clave_elector_crudo)
    
    # 4. Extraer nombre mejorado (CORREGIDO)
    nombre_completo = extraer_nombre_mejorado(textos_limpios, tipo_credencial)
    
    # 5. Extraer vigencia corregida (CORREGIDO)
    vigencia_correcta = extraer_vigencia_correcta(textos_limpios, tipo_credencial)
    
    # 6. Extraer otros campos (usar textos_limpios)
    campos: Dict[str, Any] = {
        "tipo_credencial": tipo_credencial,
        "es_ine": "INSTITUTO NACIONAL ELECTORAL" in " ".join([t.upper() for t in textos_limpios]),
        "nombre": nombre_completo,
        "curp": curp_crudo,
        "clave_elector": clave_elector_crudo,
        "fecha_nacimiento": buscar_en_lista(r'\b(\d{2}/\d{2}/\d{4})\b', textos_limpios),
        "anio_registro": buscar_en_lista(r'(\d{4}\s\d+)', textos_limpios),
        "seccion": buscar_seccion(textos_limpios),
        "vigencia": vigencia_correcta,  # Usar función corregida
        "sexo": buscar_en_lista(r'\b(H|M|X)\b', textos_limpios),
        "pais": "Mex",
    }
    
    # 7. Extraer domicilio (usar textos_limpios)
    dom_index = None
    for i, line in enumerate(textos_limpios):
        if "DOMICILIO" in line.upper():
            dom_index = i
            break
    
    if dom_index is not None:
        campos["calle"] = textos_limpios[dom_index + 1] if len(textos_limpios) > dom_index + 1 else ""
        campos["colonia"] = textos_limpios[dom_index + 2] if len(textos_limpios) > dom_index + 2 else ""
        campos["estado"] = textos_limpios[dom_index + 3] if len(textos_limpios) > dom_index + 3 else ""
    else:
        campos["calle"] = ""
        campos["colonia"] = ""
        campos["estado"] = ""
    
    # Extraer número de calle
    match_num = re.search(r'\b(\d{1,5}[A-Z]?(?:\s*INT\.?\s*\d+)?)\b', campos["calle"])
    campos["numero"] = match_num.group(1) if match_num else ""
    
    # Extraer código postal
    campos["codigo_postal"] = buscar_en_lista(r'\b(\d{5})\b', [campos["colonia"], campos["estado"]])
    
    # 8. VALIDAR Y COMPLETAR DATOS FALTANTES
    # Si falta sexo, tomarlo de la CURP
    if not campos["sexo"] and datos_curp["sexo"]:
        campos["sexo"] = datos_curp["sexo"]
    
    # Si falta fecha de nacimiento, tomarlo de la CURP
    if not campos["fecha_nacimiento"] and datos_curp["fecha_nacimiento"]:
        campos["fecha_nacimiento"] = datos_curp["fecha_nacimiento"]
    
    # Si falta sección, intentar desde clave de elector
    if not campos["seccion"] and datos_clave["seccion_clave"]:
        campos["seccion"] = datos_clave["seccion_clave"]
    
    # Si falta año de registro, intentar desde clave de elector
    if not campos["anio_registro"] and datos_clave["anio_registro_clave"]:
        campos["anio_registro"] = datos_clave["anio_registro_clave"] + " 00"
    
    # Si no hay estado del domicilio, usar el de la CURP
    if not campos["estado"] or len(campos["estado"].strip()) < 5:
        if datos_curp["estado"]:
            campos["estado"] = datos_curp["estado"]
        elif datos_clave["estado_clave"]:
            campos["estado"] = datos_clave["estado_clave"]
    
    # 9. Formatear año de registro si es necesario
    if campos["anio_registro"] and " " not in campos["anio_registro"]:
        campos["anio_registro"] = campos["anio_registro"] + " 00"
    
    # 10. Si no se encontró vigencia con la función específica, usar la búsqueda original
    if not campos["vigencia"]:
        vigencia_original = buscar_en_lista(r'(\d{4}\s*[-]?\s*?\d{4})', textos_limpios)
        if vigencia_original:
            campos["vigencia"] = vigencia_original
    
    # 11. Limpiar formato de vigencia
    if campos["vigencia"]:
        campos["vigencia"] = re.sub(r'\s+', ' ', campos["vigencia"].replace('-', ' - ').strip())
    
    return campos

# ============================================================
# 🧩 FUNCIÓN AUXILIAR: BUSCAR EN LISTA MEJORADA
# ============================================================
def buscar_en_lista(pattern: str, lista: List[str]) -> str:
    """🔍 Busca regex en lista - MEJORADA para evitar falsos positivos."""
    for line in lista:
        # Para patrones de fecha (dd/mm/yyyy), verificar que sea fecha válida
        if '\\d{2}/\\d{2}/\\d{4}' in pattern:
            match = re.search(pattern, line)
            if match:
                fecha = match.group(1)
                # Validar que sea fecha plausible
                try:
                    dia, mes, anio = map(int, fecha.split('/'))
                    if 1 <= dia <= 31 and 1 <= mes <= 12 and 1900 <= anio <= datetime.now().year:
                        return fecha
                except:
                    continue
        # Para patrones de vigencia (año - año)
        elif '\\d{4}\\s*[-]' in pattern:
            match = re.search(pattern, line)
            if match:
                vigencia = match.group(1)
                # Validar que sean años plausibles
                años = re.findall(r'\d{4}', vigencia)
                if len(años) == 2:
                    año1, año2 = int(años[0]), int(años[1])
                    if 1900 <= año1 <= 2099 and 1900 <= año2 <= 2099 and año2 > año1:
                        return vigencia
        else:
            # Para otros patrones
            match = re.search(pattern, line)
            if match:
                return match.group(1)
    
    return ""
# ============================================================
# 🧩 FUNCIONES AUXILIARES
# ============================================================
def normalizar_textos(texts: List[str]) -> List[str]:
    """🧼 Normaliza líneas OCR."""
    limpios: List[str] = []
    for t in texts:
        t2 = re.sub(r'\s+', ' ', (t or '').strip())
        if t2:
            limpios.append(t2)
    return limpios





def buscar_seccion(lista: List[str]) -> str:
    """📍 Busca sección electoral."""
    for line in lista:
        if re.fullmatch(r'\d{4}', line.strip()):
            return line.strip()
    return ""


# ============================================================
# 🪪 EXTRACCIÓN PRINCIPAL MEJORADA
# ============================================================
def extraer_campos_ine_mejorado(texts: List[str]) -> Dict[str, Any]:
    """
    🪪 Extrae campos del ANVERSO con validación desde CURP y Clave de Elector.
    """
    texts = normalizar_textos(texts)
    
    # 1. Clasificar tipo de credencial
    tipo_credencial = clasificar_tipo_credencial(texts)
    
    # 2. Extraer CURP y Clave de Elector
    curp_crudo = buscar_en_lista(r'([A-Z]{4}[0-9]{6}[HMX][A-Z]{5,6}[0-9A-Z])', texts)
    clave_elector_crudo = buscar_en_lista(r'\b([A-Z0-9]{18})\b', texts) or buscar_en_lista(r'\b([A-Z]{6}\d{8,10}[A-Z0-9]{2,4})\b', texts)
    
    # 3. Extraer datos desde CURP y Clave de Elector
    datos_curp = extraer_datos_desde_curp(curp_crudo)
    datos_clave = extraer_datos_desde_clave_elector(clave_elector_crudo)
    
    # 4. Extraer nombre mejorado
    nombre_completo = extraer_nombre_mejorado(texts, tipo_credencial)
    
    # 5. Extraer otros campos
    campos: Dict[str, Any] = {
        "tipo_credencial": tipo_credencial,
        "es_ine": "INSTITUTO NACIONAL ELECTORAL" in " ".join([t.upper() for t in texts]),
        "nombre": nombre_completo,
        "curp": curp_crudo,
        "clave_elector": clave_elector_crudo,
        "fecha_nacimiento": buscar_en_lista(r'\b(\d{2}/\d{2}/\d{4})\b', texts),
        "anio_registro": buscar_en_lista(r'(\d{4}\s\d+)', texts),
        "seccion": buscar_seccion(texts),
        "vigencia": buscar_en_lista(r'(\d{4}\s*[-]?\s*?\d{4})', texts),
        "sexo": buscar_en_lista(r'\b(H|M|X)\b', texts),
        "pais": "Mex",
    }
    
    # 6. Extraer domicilio
    dom_index = None
    for i, line in enumerate(texts):
        if "DOMICILIO" in line.upper():
            dom_index = i
            break
    
    if dom_index is not None:
        campos["calle"] = texts[dom_index + 1] if len(texts) > dom_index + 1 else ""
        campos["colonia"] = texts[dom_index + 2] if len(texts) > dom_index + 2 else ""
        campos["estado"] = texts[dom_index + 3] if len(texts) > dom_index + 3 else ""
    else:
        campos["calle"] = ""
        campos["colonia"] = ""
        campos["estado"] = ""
    
    # Extraer número de calle
    match_num = re.search(r'\b(\d{1,5}[A-Z]?(?:\s*INT\.?\s*\d+)?)\b', campos["calle"])
    campos["numero"] = match_num.group(1) if match_num else ""
    
    # Extraer código postal
    campos["codigo_postal"] = buscar_en_lista(r'\b(\d{5})\b', [campos["colonia"], campos["estado"]])
    
    # 7. VALIDAR Y COMPLETAR DATOS FALTANTES
    # Si falta sexo, tomarlo de la CURP
    if not campos["sexo"] and datos_curp["sexo"]:
        campos["sexo"] = datos_curp["sexo"]
    
    # Si falta fecha de nacimiento, tomarlo de la CURP
    if not campos["fecha_nacimiento"] and datos_curp["fecha_nacimiento"]:
        campos["fecha_nacimiento"] = datos_curp["fecha_nacimiento"]
    
    # Si falta sección, intentar desde clave de elector
    if not campos["seccion"] and datos_clave["seccion_clave"]:
        campos["seccion"] = datos_clave["seccion_clave"]
    
    # Si falta año de registro, intentar desde clave de elector
    if not campos["anio_registro"] and datos_clave["anio_registro_clave"]:
        campos["anio_registro"] = datos_clave["anio_registro_clave"] + " 00"
    
    # Si el estado del domicilio es ambiguo pero tenemos info de CURP
    if campos["estado"] and len(campos["estado"]) < 10:  # Estado muy corto o ambiguo
        if datos_curp["estado"]:
            # Verificar si el estado de la CURP es compatible
            estado_curp = datos_curp["estado"].upper()
            if any(palabra in estado_curp for palabra in campos["estado"].upper().split()):
                campos["estado"] = datos_curp["estado"]
    
    # Si no hay estado del domicilio, usar el de la CURP
    if not campos["estado"] or len(campos["estado"].strip()) < 5:
        if datos_curp["estado"]:
            campos["estado"] = datos_curp["estado"]
        elif datos_clave["estado_clave"]:
            campos["estado"] = datos_clave["estado_clave"]
    
    # 8. Formatear año de registro si es necesario
    if campos["anio_registro"] and " " not in campos["anio_registro"]:
        campos["anio_registro"] = campos["anio_registro"] + " 00"
    
    # 9. Limpiar formato de vigencia
    if campos["vigencia"]:
        campos["vigencia"] = re.sub(r'\s+', ' ', campos["vigencia"].replace('-', ' - '))
    
    return campos


# ============================================================
# 🧨 WORKER OCR CON TIMEOUT
# ============================================================
def _ocr_worker(img_bgr: np.ndarray, out_q: mp.Queue) -> None:
    """🏗️ Worker para OCR en proceso separado."""
    try:
        engine = _build_ocr_engine()
        result = engine.predict(img_bgr)
        texts = result[0]["rec_texts"] if result else []
        out_q.put({"ok": True, "texts": texts})
    except Exception as e:
        out_q.put({"ok": False, "error": str(e)})


def predict_ocr_texts_with_timeout_kill(img_bgr: np.ndarray, timeout_seconds: int) -> List[str]:
    """⏱️ OCR con timeout y kill de proceso."""
    out_q: mp.Queue = mp.Queue(maxsize=1)
    p = mp.Process(target=_ocr_worker, args=(img_bgr, out_q), daemon=True)
    
    p.start()
    p.join(timeout_seconds)
    
    if p.is_alive():
        try:
            p.terminate()
        finally:
            p.join(timeout=2)
        raise TimeoutError("OCR tardó demasiado (proceso terminado)")
    
    try:
        payload = out_q.get_nowait()
    except queue.Empty:
        raise RuntimeError("OCR terminó pero no devolvió resultado")
    
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "Error desconocido en OCR"))
    
    return payload.get("texts") or []


# ============================================================
# 🖼️ FUNCIONES DE IMAGEN
# ============================================================
def leer_imagen_desde_request(field_name: str = "imagen") -> Optional[np.ndarray]:
    """🖼️ Lee imagen del request."""
    if field_name not in request.files:
        return None
    
    file = request.files[field_name]
    data = file.read()
    if not data:
        return None
    
    npimg = np.frombuffer(data, np.uint8)
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)


# ============================================================
# 🚀 ENDPOINT OCR MEJORADO
# ============================================================
@app.route("/ocr", methods=["POST"])
def ocr_anverso_mejorado():
    """
    🪪 OCR ANVERSO MEJORADO ⭐
    ---
    tags:
      - INE OCR Mejorado
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
        description: ✅ Datos extraídos con validación desde CURP/Clave
      400:
        description: ❌ Falta imagen o imagen inválida
      408:
        description: ⏱️ OCR tardó demasiado
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
    
    # Extraer datos con validación mejorada
    datos = extraer_campos_ine_mejorado(texts)
    
    # Incluir textos OCR en modo debug
    if (request.args.get("debug") or "").strip() in ("1", "true", "True", "yes", "YES"):
        datos["_ocr_texts"] = normalizar_textos(texts)
        datos["_tipo_detectado"] = datos.get("tipo_credencial", "DESCONOCIDO")
    
    return jsonify(datos)


# ============================================================
# 🩺 HEALTH CHECK
# ============================================================
@app.route("/health", methods=["GET"])
def health_check():
    """🩺 Health Check."""
    return jsonify({
        "status": "✅ OK", 
        "service": "INE OCR API MEJORADO", 
        "version": "2.0.0",
        "features": ["Clasificación C/D/GM", "Validación CURP/Clave", "Extracción mejorada"]
    })


# ============================================================
# ▶️ RUN
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)