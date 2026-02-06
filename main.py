"""
🪪 INE/IFE OCR API MEJORADO 🇲🇽
=================================================

✅ MEJORAS IMPLEMENTADAS:
1. Clasificación automática de tipo de credencial (C, D, GH)
2. Validación y completado de datos desde CURP y Clave de Elector
3. Mejora en extracción de nombre (filtra palabras erróneas)
4. Reglas específicas por tipo de credencial
5. Mayor precisión en extracción de campos

🚀 Endpoints:
- POST /ocr  -> Procesa ANVERSO con todas las mejoras
"""

from __future__ import annotations

# ============================================================
# 🌐 MÓDULOS PRINCIPALES - FLASK + SWAGGER + CORS
# ============================================================
# 🏗️ Flask: Framework web para crear la API REST
# 📚 Flasgger: Genera documentación Swagger/OpenAPI automática
# 🔄 CORS: Permite peticiones desde otros dominios (cross-origin)
from flask import Flask, request, jsonify, send_file
from flasgger import Swagger
from flask_cors import CORS



import requests  # 🆕 Para hacer peticiones HTTP
import jwt      # 🆕 Para generar tokens JWT
from functools import wraps  # 🆕 Para decoradores

# ============================================================
# 🧠 MÓDULOS DE VISIÓN POR COMPUTADORA
# ============================================================
# 🚤 PaddleOCR: Motor de OCR principal (reconocimiento de texto en imágenes)
# 🔢 NumPy: Manipulación de arrays numéricos
# 🖼️ OpenCV: Procesamiento de imágenes
from paddleocr import PaddleOCR
import numpy as np
import cv2

# ============================================================
# 🧩 MÓDULOS UTILITARIOS
# ============================================================
# 🔍 re: Expresiones regulares para búsqueda de patrones
# 📦 io: Manejo de streams de entrada/salida
# 📝 typing: Tipado estático para mejor documentación
# 📅 datetime: Manejo de fechas y tiempos
import re
import io
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta 

# ============================================================
# 🧨 MÓDULOS PARA MANEJO DE CONCURRENCIA
# ============================================================
# 🔄 multiprocessing: Ejecución en procesos separados (para timeout)
# 🚦 queue: Comunicación entre procesos
import multiprocessing as mp
import queue


# ============================================================
# ⚙️ CONFIGURACIÓN PRINCIPAL DE FLASK
# ============================================================
# 🚀 Crea la aplicación Flask principal
app = Flask(__name__)

# 🔄 Configuración CORS (Cross-Origin Resource Sharing)
# Permite que cualquier dominio (*) acceda a la API
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",  # 🌍 Permite todos los orígenes
            "methods": ["GET", "POST", "OPTIONS"],  # 📨 Métodos HTTP permitidos
            "allow_headers": ["Content-Type", "Authorization"],  # 📋 Headers permitidos
        }
    },
)

# ============================================================
# 📚 CONFIGURACIÓN DE SWAGGER (DOCUMENTACIÓN AUTOMÁTICA)
# ============================================================
# 🎨 Plantilla de configuración para la interfaz Swagger UI
swagger_template = {
    "swagger": "2.0",  # 📖 Versión de especificación Swagger
    "info": {
        "title": "🪪 INE OCR API MEJORADO 🇲🇽",  # 🏷️ Título de la API
        "description": "API mejorada para extraer datos de credenciales INE/IFE con validación desde CURP y Clave de Elector",  # 📝 Descripción
        "version": "2.0.0",  # 🔢 Versión de la API
    },
    "basePath": "/",  # 🗺️ Ruta base de los endpoints
    "schemes": ["http"],  # 🔌 Protocolos soportados
}

# ⚙️ Configuración técnica de Swagger
swagger_config = {
    "headers": [],  # 📋 Headers adicionales
    "specs": [
        {
            "endpoint": "apispec_1",  # 🎯 Endpoint para la especificación
            "route": "/apispec_1.json",  # 🛣️ Ruta del archivo JSON
            "rule_filter": lambda rule: True,  # 🔍 Filtro de reglas (todas)
            "model_filter": lambda tag: True,  # 🏷️ Filtro de modelos (todos)
        }
    ],
    "static_url_path": "/flasgger_static",  # 📁 Ruta para archivos estáticos
    "swagger_ui": True,  # 🌐 Habilita la interfaz web de Swagger
    "specs_route": "/apidocs/",  # 🚪 Ruta de acceso a la documentación
}

# 🔧 Inicializa Swagger con la aplicación Flask
swagger = Swagger(app, template=swagger_template, config=swagger_config)


# ============================================================
# ⏱️ CONFIGURACIÓN DE TIMEOUT
# ============================================================
# ⏰ Tiempo máximo de espera para el proceso OCR (30 segundos)
OCR_TIMEOUT_SECONDS: int = 30

# ============================================================
# 📊 DICCIONARIOS DE REFERENCIA - CÓDIGOS DE ESTADO
# ============================================================
# 🗺️ Diccionario que mapea códigos de estado de 2 letras a nombres completos
# 📍 Usado para decodificar el estado de nacimiento desde la CURP
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

# 🔢 Diccionario que mapea códigos numéricos de estado a nombres completos
# 🗳️ Usado para decodificar el estado desde la Clave de Elector
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
# 🔍 CONFIGURACIÓN DEL MOTOR OCR (PADDLEOCR)
# ============================================================
def _build_ocr_engine() -> PaddleOCR:
    """🏭 Crea y configura una instancia del motor PaddleOCR.
    
    Returns:
        PaddleOCR: Instancia configurada del motor OCR
    
    Configuraciones deshabilitadas para mayor velocidad:
    - use_doc_orientation_classify: No clasifica orientación del documento
    - use_doc_unwarping: No corrige deformación de documento
    - use_textline_orientation: No corrige orientación de líneas de texto
    """
    return PaddleOCR(
        use_doc_orientation_classify=False,  # 🚫 Sin clasificación de orientación
        use_doc_unwarping=False,  # 🚫 Sin corrección de deformación
        use_textline_orientation=False,  # 🚫 Sin corrección de orientación de texto
        lang="es",  # 🇪🇸 Idioma español
    )




# ============================================================
# ⚙️ CONFIGURACIÓN JWT
# ============================================================
# 🔑 Clave secreta para firmar los JWT (cambia esto en producción)
JWT_SECRET_KEY = "clave_secreta_super_segura_cambiar_en_produccion"
# ⏰ Tiempo de expiración del token en minutos
JWT_EXPIRATION_MINUTES = 100
# 🔗 URL del API de Laravel
LARAVEL_API_URL = "https://servdes1.proyectoqroo.com.mx/gsv/ibeta/api/login"

# ============================================================
# 🔐 DECORADOR PARA AUTENTICACIÓN JWT
# ============================================================
def token_required(f):
    """🔐 Decorador para verificar tokens JWT en los endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # 🔍 Buscar token en el header Authorization
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        # 🚫 Si no hay token, retornar error
        if not token:
            return jsonify({
                "error": "❌ Token de autenticación requerido",
                "message": "Debes incluir un token JWT válido en el header Authorization: Bearer <token>"
            }), 401
        
        try:
            # 🔍 Verificar y decodificar el token
            data = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
            # 💾 Guardar información del usuario en el contexto de la petición
            request.current_user = data
        except jwt.ExpiredSignatureError:
            return jsonify({
                "error": "❌ Token expirado",
                "message": "El token ha expirado, por favor inicia sesión nuevamente"
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                "error": "❌ Token inválido",
                "message": "El token proporcionado no es válido"
            }), 401
        
        # ✅ Si todo está bien, ejecutar la función original
        return f(*args, **kwargs)
    
    return decorated

# ... (el resto de tu código existente, mantén todo igual hasta encontrar los endpoints)

# ============================================================
# 🔐 ENDPOINT DE LOGIN
# ============================================================
@app.route("/login", methods=["POST"])
def login():
    """
    🔐 ENDPOINT DE LOGIN - Autenticación contra API Laravel
    ---
    tags:
      - Autenticación
    consumes:
      - application/json
    parameters:
      - in: body
        name: credentials
        required: true
        schema:
          type: object
          required:
            - username
            - password
          properties:
            username:
              type: string
              description: 📧 Nombre de usuario
              example: "gsvopb"
            password:
              type: string
              description: 🔑 Contraseña
              example: "gsvopb"
    responses:
      200:
        description: ✅ Login exitoso, retorna tokens JWT y Laravel
      401:
        description: ❌ Credenciales incorrectas
      500:
        description: ⚠️ Error al conectar con el servidor de autenticación
    """
    # 📥 Obtener credenciales del request
    data = request.get_json()
    
    # 🚫 Validar que se enviaron credenciales
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({
            "error": "❌ Credenciales incompletas",
            "message": "Debes enviar username y password"
        }), 400
    
    username = data['username']
    password = data['password']
    
    # 🚀 Preparar payload para API Laravel
    laravel_payload = {
        "username": username,
        "password": password
    }
    
    try:
        # 🔗 Hacer petición POST a la API de Laravel
        response = requests.post(
            LARAVEL_API_URL,
            json=laravel_payload,
            timeout=10  # ⏰ Timeout de 10 segundos
        )
        
        # 🔍 Analizar respuesta de Laravel
        if response.status_code == 200:
            laravel_data = response.json()
            
            # 📝 Verificar estructura de respuesta esperada
            if 'token' in laravel_data and 'user' in laravel_data:
                # 🎯 Crear payload para JWT
                jwt_payload = {
                    "user_id": laravel_data['user']['id'],
                    "username": laravel_data['user']['username'],
                    "nombre": laravel_data['user']['nombre'],
                    # ⏰ Agregar fecha de expiración (100 minutos)
                    "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES),
                    # 📅 Agregar fecha de emisión
                    "iat": datetime.utcnow()
                }
                
                # 🔐 Generar token JWT
                jwt_token = jwt.encode(jwt_payload, JWT_SECRET_KEY, algorithm="HS256")
                
                # ✅ Retornar respuesta exitosa
                return jsonify({
                    "message": "✅ Autenticación exitosa",
                    "token": jwt_token,  # 🔐 Token JWT generado por nosotros
                    "token_laravel": laravel_data['token'],  # 🔗 Token original de Laravel
                    "user": laravel_data['user'],  # 👤 Información del usuario
                    "expires_in": JWT_EXPIRATION_MINUTES * 60  # ⏳ Tiempo de expiración en segundos
                }), 200
            else:
                # ⚠️ Respuesta inesperada de Laravel
                return jsonify({
                    "error": "⚠️ Respuesta inesperada del servidor",
                    "message": "La respuesta del servidor no contiene la estructura esperada"
                }), 500
                
        elif response.status_code == 401:
            # ❌ Credenciales incorrectas
            error_data = response.json()
            return jsonify({
                "error": "❌ Credenciales incorrectas",
                "message": error_data.get('message', 'Usuario o contraseña incorrectos')
            }), 401
            
        else:
            # ⚠️ Otro error del servidor Laravel
            return jsonify({
                "error": f"⚠️ Error del servidor (Código: {response.status_code})",
                "message": "Error al autenticar con el servidor remoto"
            }), response.status_code
            
    except requests.exceptions.Timeout:
        # ⏰ Timeout en la conexión
        return jsonify({
            "error": "⏰ Timeout de conexión",
            "message": "El servidor de autenticación no responde"
        }), 504
        
    except requests.exceptions.ConnectionError:
        # 🔌 Error de conexión
        return jsonify({
            "error": "🔌 Error de conexión",
            "message": "No se puede conectar con el servidor de autenticación"
        }), 503
        
    except Exception as e:
        # ❌ Error general
        return jsonify({
            "error": "❌ Error interno",
            "message": f"Error al procesar la autenticación: {str(e)}"
        }), 500

# ============================================================
# 🔐 ENDPOINT VERIFY TOKEN
# ============================================================
@app.route("/verify-token", methods=["GET"])
@token_required  # 🔐 Requiere token válido
def verify_token():
    """
    🔍 ENDPOINT PARA VERIFICAR TOKEN
    ---
    tags:
      - Autenticación
    security:
      - BearerAuth: []
    parameters:
      - name: Authorization
        in: header
        type: string
        required: true
        description: 🔐 Token JWT en formato "Bearer {token}"
    responses:
      200:
        description: ✅ Token válido con información del usuario
      401:
        description: ❌ Token inválido o expirado
    """
    current_user = getattr(request, 'current_user', {})
    
    # ⏰ Calcular tiempo restante
    exp_timestamp = current_user.get('exp', 0)
    if exp_timestamp:
        exp_datetime = datetime.fromtimestamp(exp_timestamp)
        remaining = exp_datetime - datetime.utcnow()
        remaining_minutes = max(0, int(remaining.total_seconds() / 60))
    else:
        remaining_minutes = 0
    
    return jsonify({
        "message": "✅ Token válido",
        "user": {
            "user_id": current_user.get('user_id'),
            "username": current_user.get('username'),
            "nombre": current_user.get('nombre')
        },
        "token_valid": True,
        "remaining_minutes": remaining_minutes,
        "expires_at": exp_timestamp
    }), 200

# ... (el resto de tu código existente)

# ============================================================
# 🔐 ACTUALIZAR CONFIGURACIÓN DE SWAGGER PARA INCLUIR SECURITY
# ============================================================
swagger_template = {
    "swagger": "2.0",  # 📖 Versión de especificación Swagger
    "info": {
        "title": "🪪 INE OCR API MEJORADO 🇲🇽",  # 🏷️ Título de la API
        "description": "API mejorada para extraer datos de credenciales INE/IFE con validación desde CURP y Clave de Elector\n\n## 🔐 Autenticación\n\nEsta API requiere autenticación JWT. Para usar los endpoints protegidos:\n\n1. Primero obtén un token en `/login`\n2. Incluye el token en el header: `Authorization: Bearer {token}`",  # 📝 Descripción actualizada
        "version": "2.0.0",  # 🔢 Versión de la API
    },
    "basePath": "/",  # 🗺️ Ruta base de los endpoints
    "schemes": ["http"],  # 🔌 Protocolos soportados
    "securityDefinitions": {  # 🆕 Definiciones de seguridad
        "BearerAuth": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "🔐 Ingresa tu token JWT en el formato: Bearer {token}"
        }
    },
    "security": [  # 🆕 Seguridad por defecto (opcional)
        {
            "BearerAuth": []
        }
    ]
}


# ============================================================
# 🏷️ CLASIFICACIÓN DE TIPO DE CREDENCIAL
# ============================================================
def clasificar_tipo_credencial(textos_limpios: List[str]) -> str:
    """
    🪪 Clasifica automáticamente el tipo de credencial INE/IFE.
    
    🎯 Tipos posibles:
    - "C": Credencial IFE antigua (Instituto Federal Electoral)
    - "D": Credencial INE estándar
    - "GH": Credencial INE con clave de elector
    
    🔍 Estrategia de clasificación:
    1. Primero detecta IFE (tipo C) por palabras clave específicas
    2. Luego diferencia entre D y GH por presencia de "CLAVE DE ELECTOR"
    
    Args:
        textos_limpios (List[str]): Lista de textos extraídos por OCR
        
    Returns:
        str: "C", "D" o "GH"
    """
    # 📝 Unifica todos los textos en uno solo para búsqueda más fácil
    texto_completo = " ".join([t.upper().strip() for t in textos_limpios if t]).strip()

    # ============================================================
    # ✅ 1) DETECCIÓN DE TIPO C (IFE ANTIGUA)
    # ============================================================
    # 🔎 Busca indicadores específicos de credenciales IFE
    es_ife = (
        "INSTITUTO FEDERAL ELECTORAL" in texto_completo  # 🏛️ Nombre completo del IFE
        or "REGISTRO FEDERAL DE ELECTORES" in texto_completo  # 📋 Texto característico
        or re.search(r"\bIFE\b", texto_completo) is not None  # 🔠 Siglas IFE
        or ("FEDERAL" in texto_completo and "ELECTORAL" in texto_completo and "REGISTRO" in texto_completo)  # 🧩 Combinación de palabras
    )

    if es_ife:
        return "C"  # ✅ Retorna tipo C (IFE)

    # ============================================================
    # ✅ 2) DETECCIÓN DE CREDENCIALES INE (D O GH)
    # ============================================================
    # 🔍 Verifica si es una credencial INE (Instituto Nacional Electoral)
    tiene_ine = (
        ("INSTITUTO" in texto_completo and "ELECTORAL" in texto_completo)  # 🏢 "INSTITUTO" + "ELECTORAL"
        and ("NACIONAL" in texto_completo or re.search(r"\bINE\b", texto_completo) is not None)  # 🇲🇽 "NACIONAL" o siglas INE
    )

    # 📄 Verifica si es una "CREDENCIAL PARA VOTAR"
    tiene_credencial_para_votar = "CREDENCIAL" in texto_completo and "VOTAR" in texto_completo

    # 🔤 Busca CURP en el texto (patrón específico)
    tiene_curp = (
        "CURP" in texto_completo  # 📛 Texto "CURP"
        or re.search(r'\b[A-Z]{4}\d{6}[HMX][A-Z]{5,6}[0-9A-Z]\b', texto_completo) is not None  # 🧬 Patrón de CURP
    )

    # 🔑 Busca "CLAVE DE ELECTOR" con flexibilidad (OCR puede tener errores)
    tiene_clave_elector_flexible = (
        re.search(r'CLAVE\s*DE\s*ELECTOR', texto_completo) is not None  # 🔍 Regex flexible
        or ("CLAVE" in texto_completo and "ELECTOR" in texto_completo)  # 🧩 Ambas palabras
        or re.search(r'CLAVE\s*DE\s*ELEC', texto_completo) is not None  # 🔠 Variación corta
    )

    # ============================================================
    # ✅ 3) CLASIFICACIÓN FINAL INE (GH vs D)
    # ============================================================
    # 🎯 Tipo GH: INE + Credencial para votar + Clave de elector
    if tiene_ine and tiene_credencial_para_votar and tiene_clave_elector_flexible:
        return "GH"  # ✅ Tipo GH (con clave de elector)

    # 🎯 Tipo D: INE + Credencial para votar (sin clave de elector clara)
    if tiene_ine and tiene_credencial_para_votar:
        return "D"  # ✅ Tipo D (estándar)

    # ⚠️ Default: Si no se clasifica, asume tipo D
    return "D"


# ============================================================
# 🧠 VALIDACIÓN Y EXTRACCIÓN DESDE CURP
# ============================================================
def extraer_datos_desde_curp(curp: str) -> Dict[str, str]:
    """
    📊 Extrae información demográfica validada desde una CURP.
    
    🧬 Estructura de la CURP (18 caracteres):
    - Posiciones 1-4: Letras iniciales apellidos y nombre
    - Posiciones 5-10: Fecha de nacimiento (AAMMDD)
    - Posición 11: Sexo (H/M)
    - Posiciones 12-13: Entidad federativa de nacimiento
    - Posiciones 14-16: Consonantes internas
    - Posición 17: Diferencia entre nombres similares
    - Posición 18: Dígito verificador
    
    Args:
        curp (str): CURP extraída del texto OCR
        
    Returns:
        Dict[str, str]: Diccionario con datos extraídos:
            - sexo: "H", "M" o "X"
            - fecha_nacimiento: Formato DD/MM/YYYY
            - entidad_nacimiento: Código de 2 letras
            - estado: Nombre completo del estado
    """
    # 📦 Diccionario inicial con valores vacíos
    datos = {
        "sexo": "",
        "fecha_nacimiento": "",
        "entidad_nacimiento": "",
        "estado": ""
    }
    
    # 🚫 Validación: CURP debe tener al menos 16 caracteres
    if not curp or len(curp) < 16:
        return datos
    
    # 1. 🔍 EXTRACCIÓN DE SEXO (10º carácter, índice 10)
    if len(curp) >= 10:
        sexo_char = curp[10].upper()  # 📍 Carácter en posición 10 (0-indexed)
        if sexo_char == 'H':
            datos["sexo"] = "H"  # 👨 Masculino
        elif sexo_char == 'M':
            datos["sexo"] = "M"  # 👩 Femenino
        else:
            datos["sexo"] = "X"  # ❓ No especificado
    
    # 2. 📅 EXTRACCIÓN DE FECHA DE NACIMIENTO (posiciones 5-10: AAMMDD)
    if len(curp) >= 10:
        anio = curp[4:6]  # 🗓️ Últimos 2 dígitos del año (posiciones 5-6)
        mes = curp[6:8]   # 📅 Mes (posiciones 7-8)
        dia = curp[8:10]  # 📆 Día (posiciones 9-10)
        
        # 🤔 Determinación del siglo (1900s o 2000s)
        año_actual_2dig = datetime.now().year % 100  # 🎯 Últimos 2 dígitos del año actual
        año_num = int(anio)  # 🔢 Convierte a número
        
        # 🕰️ Si el año extraído es mayor al año actual, asume siglo 19, sino 20
        siglo = "19" if año_num > año_actual_2dig else "20"
        
        # 🗓️ Formatea fecha completa DD/MM/YYYY
        datos["fecha_nacimiento"] = f"{dia}/{mes}/{siglo}{anio}"
    
    # 3. 🗺️ EXTRACCIÓN DE ENTIDAD DE NACIMIENTO (posiciones 12-13)
    if len(curp) >= 13:
        codigo_estado = curp[11:13].upper()  # 📍 Código de 2 letras (posiciones 12-13)
        datos["entidad_nacimiento"] = codigo_estado  # 🔤 Código (ej: "DF")
        datos["estado"] = CODIGOS_ESTADO_CURP.get(codigo_estado, "")  # 🏙️ Nombre completo
    
    return datos


# ============================================================
# 🗳️ VALIDACIÓN Y EXTRACCIÓN DESDE CLAVE DE ELECTOR
# ============================================================
def extraer_datos_desde_clave_elector(clave: str) -> Dict[str, str]:
    """
    📍 Extrae información geográfica y temporal desde la Clave de Elector.
    
    🔑 Estructura típica de Clave de Elector (18-19 caracteres):
    - Posiciones 1-2: Código del estado (01-32)
    - Posiciones 3-6: Municipio
    - Posiciones 7-10: Sección electoral
    - Posiciones 11-14: Año de registro
    - Posiciones 15-18: Número consecutivo
    
    Args:
        clave (str): Clave de elector extraída del texto OCR
        
    Returns:
        Dict[str, str]: Diccionario con datos extraídos:
            - estado_clave: Nombre del estado
            - seccion_clave: Sección electoral (4 dígitos)
            - anio_registro_clave: Año de registro
    """
    # 📦 Diccionario inicial con valores vacíos
    datos = {
        "estado_clave": "",
        "seccion_clave": "",
        "anio_registro_clave": ""
    }
    
    # 🚫 Validación: Clave debe tener al menos 13 caracteres
    if not clave or len(clave) < 13:
        return datos
    
    # 1. 🗺️ EXTRACCIÓN DEL ESTADO (primeros 2 dígitos)
    if len(clave) >= 2:
        codigo_estado = clave[0:2]  # 🔢 Primeros 2 caracteres
        datos["estado_clave"] = CODIGOS_ESTADO_ELECTOR.get(codigo_estado, "")  # 🏙️ Nombre del estado
    
    # 2. 📍 EXTRACCIÓN DE SECCIÓN ELECTORAL
    # 🔍 Busca 4 dígitos consecutivos que representen la sección
    seccion_match = re.search(r'\b(\d{4})\b', clave)
    if seccion_match:
        datos["seccion_clave"] = seccion_match.group(1)  # ✅ 4 dígitos encontrados
    
    # 3. 📅 EXTRACCIÓN DE AÑO DE REGISTRO
    # 🔎 Busca patrones de 4 dígitos que sean años plausibles (1900-2025)
    for match in re.finditer(r'\b(19\d{2}|20[0-2]\d)\b', clave):
        año = int(match.group())  # 🔢 Convierte a número
        # ✅ Valida que sea un año razonable
        if 1900 <= año <= datetime.now().year + 1:
            datos["anio_registro_clave"] = str(año)  # 🗓️ Año válido encontrado
            break  # ⏹️ Solo toma el primer año válido
    
    return datos


# ============================================================
# 👤 MEJORA EN EXTRACCIÓN DE NOMBRE
# ============================================================
def limpiar_y_validar_nombre(nombre: str) -> str:
    """
    🧹 Limpia y valida un nombre extraído por OCR.
    
    🚫 Elimina palabras que NO deberían estar en un nombre:
    - Términos administrativos ("EDAD", "AÑOS", "DOMICILIO")
    - Palabras relacionadas con la credencial ("CURP", "CLAVE")
    - Números y códigos
    
    Args:
        nombre (str): Nombre crudo extraído por OCR
        
    Returns:
        str: Nombre limpio y validado
    """
    if not nombre:
        return ""  # 🚫 Retorna vacío si no hay nombre
    
    # 🚫 LISTA DE PALABRAS INVÁLIDAS EN NOMBRES
    palabras_invalidas = [
        'EDAD', 'AÑOS', 'AÑO', 'EDAD:', 'EDADES', 'FECHA', 'NACIMIENTO',
        'DOMICILIO', 'CALLE', 'COLONIA', 'ESTADO', 'MUNICIPIO', 'CIUDAD',
        'CP', 'C.P.', 'CÓDIGO', 'POSTAL', 'SECCIÓN', 'SECCION', 'CLAVE',
        'ELECTOR', 'CURP', 'VIGENCIA', 'VIGENTE', 'INSTITUTO', 'NACIONAL',
        'FEDERAL', 'ELECTORAL', 'CREDENCIAL', 'VOTAR', 'PARA', 'MÉXICO',
        'REGISTRO'  # ✅ Evita "DE REGISTRO" en nombres
    ]
    
    # 🔠 Convierte a mayúsculas para comparación sin case-sensitive
    nombre_upper = nombre.upper()
    
    # 🧩 Separa el nombre en palabras individuales
    palabras = nombre_upper.split()
    palabras_limpias = []  # 📦 Lista para palabras válidas
    
    for palabra in palabras:
        # 🧼 Limpia caracteres no alfabéticos (mantiene Ñ y tildes)
        palabra_limpia = re.sub(r'[^\wÁÉÍÓÚÜÑ]', '', palabra)
        
        # ✅ CRITERIOS DE VALIDACIÓN:
        # 1. No vacía
        # 2. Más de 1 carácter
        # 3. No está en la lista de palabras inválidas
        # 4. No es solo dígitos
        # 5. No es patrón mixto de números y letras
        if (palabra_limpia and 
            len(palabra_limpia) > 1 and 
            palabra_limpia not in palabras_invalidas and
            not palabra_limpia.isdigit() and
            not re.match(r'^\d+[A-Z]*$', palabra_limpia)):
            palabras_limpias.append(palabra)  # ✅ Palabra válida
    
    # 🔄 Reconstruye el nombre manteniendo la capitalización original
    nombre_original = nombre.split()  # 🧩 Palabras con formato original
    nombre_final = []  # 📦 Nombre final reconstruido
    
    for palabra in nombre_original:
        # 🔍 Verifica si la palabra (en mayúsculas) está en las palabras limpias
        if palabra.upper() in [p.upper() for p in palabras_limpias]:
            nombre_final.append(palabra)  # ✅ Mantiene formato original
    
    return " ".join(nombre_final)  # 🔗 Une palabras con espacios


# ============================================================
# 👤 CORRECCIÓN: EXTRACCIÓN DE NOMBRE PARA TIPO GH
# ============================================================
def extraer_nombre_mejorado(texts: List[str], tipo_credencial: str) -> str:
    """
    👤 Extrae el nombre completo desde textos OCR con estrategias específicas.
    
    🎯 Estrategias implementadas:
    1. 🏠 ANCLA POR "DOMICILIO": Busca nombre arriba de la palabra "DOMICILIO"
    2. 🏷️ BUSQUEDA POR "NOMBRE": Para tipo GH, busca etiqueta "NOMBRE"
    3. 🔄 FALLBACK GENERAL: Búsqueda heurística general
    
    ✅ FIX IMPORTANTE: Maneja casos donde OCR pega "EDAD" al nombre
    
    Args:
        texts (List[str]): Lista de textos extraídos por OCR
        tipo_credencial (str): "C", "D" o "GH"
        
    Returns:
        str: Nombre completo extraído y limpiado
    """
    # 🧼 Normaliza los textos (elimina espacios múltiples, etc.)
    textos_limpios = normalizar_textos(texts)

    # 🚫 EXPRESIONES REGULARES PARA FILTRAR
    blacklist_regex = r'(INSTITUTO|NACIONAL|ELECTORAL|CREDENCIAL|PARA\s+VOTAR|M[EÉ]XICO|ESTADOS\s+UNIDOS)'
    # 🛑 STOP LABELS: Palabras que indican fin del nombre
    stop_labels_regex = r'(DOMICILIO|CLAVE|CURP|FECHA|SECCI[ÓO]N|AÑO|REGISTRO|VIGENCIA|SEXO|EDAD)'

    # ============================================================
    # ✅ ESTRATEGIA 0: ANCLA POR "DOMICILIO" (UNIVERSAL)
    # ============================================================
    # 🎯 Busca la palabra "DOMICILIO" como punto de referencia
    idx_dom = None
    for i, line in enumerate(textos_limpios):
        if "DOMICILIO" in line.upper():
            idx_dom = i  # 📍 Índice donde aparece "DOMICILIO"
            break

    if idx_dom is not None:
        # 🔍 Busca en las 12 líneas anteriores a "DOMICILIO"
        ventana = textos_limpios[max(0, idx_dom - 12):idx_dom]
        candidatos = []  # 📦 Lista de candidatos a nombre

        for s in ventana:
            s = s.strip()  # 🧼 Limpia espacios
            up = s.upper().strip()  # 🔠 Versión mayúsculas

            if not s:  # 🚫 Ignora vacíos
                continue
            if re.fullmatch(r'NOMBRE', up):  # 🚫 Ignora solo "NOMBRE"
                continue
            if re.search(stop_labels_regex, up):  # 🛑 Para en stop labels
                continue
            if re.search(blacklist_regex, up):  # 🚫 Filtra blacklist
                continue
            if any(ch.isdigit() for ch in up):  # 🔢 Filtra números
                continue
            # 🚫 Ignora líneas muy cortas (probablemente ruido)
            if len(re.sub(r'[^A-ZÁÉÍÓÚÜÑ]', '', up)) < 2:
                continue

            candidatos.append(s)  # ✅ Agrega candidato válido

        # 🎯 Toma las últimas 2-4 líneas como nombre completo
        if candidatos:
            nombre_candidato = " ".join(candidatos[-4:]).strip()
            # 🧼 Limpia y valida el nombre
            nombre_candidato = limpiar_y_validar_nombre(nombre_candidato).strip()

            # ✅ Requiere al menos 2 palabras para ser válido
            if len(nombre_candidato.split()) >= 2:
                return nombre_candidato

    # ============================================================
    # 🪪 ESTRATEGIA ESPECÍFICA PARA TIPO GH
    # ============================================================
    if tipo_credencial == "GH":
        # 🔍 Busca línea que solo diga "NOMBRE"
        for i, line in enumerate(textos_limpios):
            up = line.upper().strip()

            if re.fullmatch(r'^NOMBRE\s*$', up):
                partes: List[str] = []  # 📦 Partes del nombre

                # 🔍 Busca en las siguientes 7 líneas después de "NOMBRE"
                for j in range(i + 1, min(i + 7, len(textos_limpios))):
                    s = textos_limpios[j].strip()
                    s_up = s.upper().strip()

                    if re.search(stop_labels_regex, s_up):  # 🛑 Stop label
                        break
                    if re.search(blacklist_regex, s_up):  # 🚫 Blacklist
                        continue
                    if not s:  # 🚫 Vacío
                        continue
                    if any(ch.isdigit() for ch in s_up):  # 🔢 Números
                        continue
                    # 🚫 Texto muy corto
                    if len(re.sub(r'[^A-ZÁÉÍÓÚÜÑ]', '', s_up)) < 2:
                        continue

                    partes.append(s)  # ✅ Parte válida del nombre

                # 🔗 Une las partes y limpia
                nombre_candidato = " ".join(partes).strip()
                nombre_candidato = limpiar_y_validar_nombre(nombre_candidato).strip()

                # ✅ Requiere al menos 2 palabras
                if len(nombre_candidato.split()) >= 2:
                    return nombre_candidato

        # 🔍 Busca "NOMBRE: ..." en la misma línea
        for line in textos_limpios:
            up = line.upper()
            # 🎯 Regex para "NOMBRE:" seguido del nombre
            m = re.search(r'NOMBRE\s*[:\-]?\s*([A-ZÁÉÍÓÚÜÑ\s\.]{3,})', up)
            if m:
                nombre_candidato = m.group(1).strip()
                nombre_candidato = limpiar_y_validar_nombre(nombre_candidato).strip()

                nc_up = nombre_candidato.upper()
                # ✅ Validaciones múltiples
                if (
                    len(nombre_candidato.split()) >= 2
                    and not re.search(stop_labels_regex, nc_up)
                    and not re.search(blacklist_regex, nc_up)
                    and not any(ch.isdigit() for ch in nc_up)
                ):
                    return nombre_candidato

    # ============================================================
    # 🔄 ESTRATEGIA FALLBACK GENERAL
    # ============================================================
    candidatos = []  # 📦 Candidatos encontrados
    for line in textos_limpios:
        up = line.upper().strip()
        if not up:  # 🚫 Vacío
            continue
        if len(up.split()) < 2:  # 🚫 Menos de 2 palabras
            continue
        if re.search(stop_labels_regex, up):  # 🛑 Stop label
            continue
        if re.search(blacklist_regex, up):  # 🚫 Blacklist
            continue
        if any(ch.isdigit() for ch in up):  # 🔢 Números
            continue

        # 🧼 Limpia y valida candidato
        candidato = limpiar_y_validar_nombre(line.strip()).strip()
        if len(candidato.split()) >= 2:  # ✅ Al menos 2 palabras
            candidatos.append(candidato)

    # 🎯 Retorna el primer candidato válido
    if candidatos:
        return candidatos[0]

    return ""  # 🚫 Sin nombre encontrado


# ============================================================
# 📅 CORRECCIÓN: EXTRACCIÓN DE VIGENCIA
# ============================================================
def extraer_vigencia_correcta(texts: List[str], tipo_credencial: str) -> str:
    """
    📅 Extrae correctamente el período de vigencia de la credencial.
    
    🎯 Maneja formatos comunes:
    - "2021 - 2031"
    - "VIGENCIA: 2021-2031"
    - "VIGENCIA 2021 2031"
    
    Args:
        texts (List[str]): Lista de textos extraídos por OCR
        tipo_credencial (str): Tipo de credencial (no usado aquí pero mantenido)
        
    Returns:
        str: Período de vigencia en formato "AAAA - AAAA"
    """
    # 🧼 Normaliza textos
    textos_limpios = normalizar_textos(texts)
    
    # 🔍 BUSQUEDA POR PATRÓN "VIGENCIA" EXPLÍCITO
    for line in textos_limpios:
        line_upper = line.upper()
        
        # 🎯 Busca línea que contenga "VIGENCIA"
        if "VIGENCIA" in line_upper:
            # 🔍 Intenta extraer de la misma línea: "VIGENCIA: 2021-2031"
            match = re.search(r'VIGENCIA\s*[:\-]?\s*(\d{4}\s*[-\s]+\s*\d{4})', line_upper)
            if match:
                vigencia = match.group(1)
                # 🧼 Limpia formato: estandariza espacios y guiones
                vigencia = re.sub(r'\s+', ' ', vigencia.replace('-', ' - ').strip())
                return vigencia  # ✅ Vigencia encontrada
            
            # 🔍 Si no está en la misma línea, busca en líneas siguientes
            idx = textos_limpios.index(line)
            for j in range(idx + 1, min(idx + 3, len(textos_limpios))):
                siguiente = textos_limpios[j]
                # 🎯 Busca patrón de dos años con guión
                match = re.search(r'(\d{4}\s*[-\s]+\s*\d{4})', siguiente)
                if match:
                    vigencia = match.group(1)
                    # 🧼 Limpia formato
                    vigencia = re.sub(r'\s+', ' ', vigencia.replace('-', ' - ').strip())
                    return vigencia  # ✅ Vigencia encontrada
        
        # 🔍 BUSQUEDA DIRECTA DE PATRÓN DE AÑOS CON GUION
        # 🎯 Busca "2021-2031" directamente en cualquier línea
        match = re.search(r'\b(\d{4}\s*[-]\s*\d{4})\b', line)
        if match:
            # ✅ Valida que sean años plausibles
            años = re.findall(r'\d{4}', match.group(1))
            if len(años) == 2:
                año1, año2 = int(años[0]), int(años[1])
                # 🕰️ Rango válido: 1900-2099 y año2 > año1
                if 1900 <= año1 <= 2099 and 1900 <= año2 <= 2099 and año2 > año1:
                    vigencia = match.group(1)
                    # 🧼 Limpia formato
                    vigencia = re.sub(r'\s+', ' ', vigencia.replace('-', ' - ').strip())
                    return vigencia  # ✅ Vigencia válida
    
    # 🔍 BUSQUEDA POR "VIGENCIA" SEGUIDO DE AÑOS SEPARADOS
    for i, line in enumerate(textos_limpios):
        if "VIGENCIA" in line.upper():
            # 🔍 Revisa las próximas 3 líneas
            for j in range(i, min(i + 3, len(textos_limpios))):
                siguiente = textos_limpios[j]
                # 🎯 Busca cualquier patrón de año (1900-2099)
                años = re.findall(r'\b(19\d{2}|20\d{2})\b', siguiente)
                if len(años) >= 2:
                    return f"{años[0]} - {años[1]}"  # ✅ Dos años encontrados
                elif len(años) == 1 and j > i:
                    # 🔍 Si solo hay un año, busca el segundo en siguiente línea
                    siguiente2 = textos_limpios[j + 1] if j + 1 < len(textos_limpios) else ""
                    año2_match = re.search(r'\b(19\d{2}|20\d{2})\b', siguiente2)
                    if año2_match:
                        return f"{años[0]} - {año2_match.group(1)}"  # ✅ Segundo año encontrado
    
    return ""  # 🚫 Sin vigencia encontrada


# ============================================================
# 🪪 FUNCIÓN PRINCIPAL CORREGIDA
# ============================================================
def extraer_campos_ine_mejorado(texts: List[str]) -> Dict[str, Any]:
    """
    🪪 Función principal que extrae y valida todos los campos del ANVERSO.
    
    🎯 Flujo de procesamiento:
    1. 📝 Clasifica tipo de credencial (C/D/GH)
    2. 🔍 Extrae CURP y Clave de Elector
    3. 🧠 Valida datos desde CURP y Clave
    4. 👤 Extrae nombre mejorado
    5. 📅 Extrae vigencia corregida
    6. 🏠 Extrae domicilio y otros campos
    7. ✅ Completa datos faltantes con validación
    
    Args:
        texts (List[str]): Lista de textos extraídos por OCR
        
    Returns:
        Dict[str, Any]: Diccionario con todos los campos extraídos
    """
    # 🧼 1. NORMALIZACIÓN INICIAL
    textos_limpios = normalizar_textos(texts)
    
    # 🏷️ 2. CLASIFICACIÓN DE TIPO DE CREDENCIAL
    tipo_credencial = clasificar_tipo_credencial(textos_limpios)
    
    # 🔍 3. EXTRACCIÓN DE CURP Y CLAVE DE ELECTOR
    curp_crudo = buscar_en_lista(r'([A-Z]{4}[0-9]{6}[HMX][A-Z]{5,6}[0-9A-Z])', textos_limpios)
    clave_elector_crudo = buscar_en_lista(r'\b([A-Z0-9]{18})\b', textos_limpios) or buscar_en_lista(r'\b([A-Z]{6}\d{8,10}[A-Z0-9]{2,4})\b', textos_limpios)
    
    # 🧠 4. VALIDACIÓN DESDE CURP Y CLAVE
    datos_curp = extraer_datos_desde_curp(curp_crudo)
    datos_clave = extraer_datos_desde_clave_elector(clave_elector_crudo)
    
    # 👤 5. EXTRACCIÓN DE NOMBRE MEJORADO (CORREGIDO)
    nombre_completo = extraer_nombre_mejorado(textos_limpios, tipo_credencial)
    
    # 📅 6. EXTRACCIÓN DE VIGENCIA CORREGIDA
    vigencia_correcta = extraer_vigencia_correcta(textos_limpios, tipo_credencial)
    
    # 📦 7. EXTRACCIÓN DE OTROS CAMPOS BÁSICOS
    campos: Dict[str, Any] = {
        "tipo_credencial": tipo_credencial,  # 🏷️ C, D o GH
        "es_ine": "INSTITUTO NACIONAL ELECTORAL" in " ".join([t.upper() for t in textos_limpios]),  # 🇲🇽 Es INE (no IFE)
        "nombre": nombre_completo,  # 👤 Nombre completo
        "curp": curp_crudo,  # 🧬 CURP cruda
        "clave_elector": clave_elector_crudo,  # 🔑 Clave de elector cruda
        "fecha_nacimiento": buscar_en_lista(r'\b(\d{2}/\d{2}/\d{4})\b', textos_limpios),  # 📅 Fecha DD/MM/YYYY
        "anio_registro": buscar_en_lista(r'(\d{4}\s\d+)', textos_limpios),  # 🗓️ Año registro + código
        "seccion": buscar_seccion(textos_limpios),  # 📍 Sección electoral
        "vigencia": vigencia_correcta,  # 📅 Período de vigencia
        "sexo": buscar_en_lista(r'\b(H|M|X)\b', textos_limpios),  # 👫 Sexo
        "pais": "Mex",  # 🇲🇽 País por defecto
    }
    
    # 🏠 8. EXTRACCIÓN DE DOMICILIO
    dom_index = None
    for i, line in enumerate(textos_limpios):
        if "DOMICILIO" in line.upper():
            dom_index = i  # 📍 Índice de "DOMICILIO"
            break
    
    # 🏡 Asigna líneas después de "DOMICILIO" a campos de dirección
    if dom_index is not None:
        campos["calle"] = textos_limpios[dom_index + 1] if len(textos_limpios) > dom_index + 1 else ""  # 🛣️ Calle
        campos["colonia"] = textos_limpios[dom_index + 2] if len(textos_limpios) > dom_index + 2 else ""  # 🏘️ Colonia
        campos["estado"] = textos_limpios[dom_index + 3] if len(textos_limpios) > dom_index + 3 else ""  # 🏙️ Estado
    else:
        campos["calle"] = ""
        campos["colonia"] = ""
        campos["estado"] = ""
    
    # 🔢 9. EXTRACCIÓN DE NÚMERO DE CALLE
    # 🎯 Busca número con posibles sufijos como "INT. 1"
    match_num = re.search(r'\b(\d{1,5}[A-Z]?(?:\s*INT\.?\s*\d+)?)\b', campos["calle"])
    campos["numero"] = match_num.group(1) if match_num else ""  # 🏷️ Número extraído
    
    # 📮 10. EXTRACCIÓN DE CÓDIGO POSTAL
    campos["codigo_postal"] = buscar_en_lista(r'\b(\d{5})\b', [campos["colonia"], campos["estado"]])  # 🔢 5 dígitos
    
    # ============================================================
    # ✅ 11. VALIDACIÓN Y COMPLETADO DE DATOS FALTANTES
    # ============================================================
    
    # 👫 Si falta sexo, tomarlo de la CURP
    if not campos["sexo"] and datos_curp["sexo"]:
        campos["sexo"] = datos_curp["sexo"]
    
    # 📅 Si falta fecha de nacimiento, tomarlo de la CURP
    if not campos["fecha_nacimiento"] and datos_curp["fecha_nacimiento"]:
        campos["fecha_nacimiento"] = datos_curp["fecha_nacimiento"]
    
    # 📍 Si falta sección, intentar desde clave de elector
    if not campos["seccion"] and datos_clave["seccion_clave"]:
        campos["seccion"] = datos_clave["seccion_clave"]
    
    # 🗓️ Si falta año de registro, intentar desde clave de elector
    if not campos["anio_registro"] and datos_clave["anio_registro_clave"]:
        campos["anio_registro"] = datos_clave["anio_registro_clave"] + " 00"  # 🔢 Agrega "00" como código
    
    # 🏙️ Si no hay estado del domicilio, usar el de la CURP o Clave
    if not campos["estado"] or len(campos["estado"].strip()) < 5:
        if datos_curp["estado"]:
            campos["estado"] = datos_curp["estado"]  # 🗺️ Estado desde CURP
        elif datos_clave["estado_clave"]:
            campos["estado"] = datos_clave["estado_clave"]  # 🗺️ Estado desde Clave
    
    # 🔢 12. FORMATEAR AÑO DE REGISTRO (agregar " 00" si falta)
    if campos["anio_registro"] and " " not in campos["anio_registro"]:
        campos["anio_registro"] = campos["anio_registro"] + " 00"
    
    # 📅 13. FALLBACK PARA VIGENCIA (si la función específica no encontró)
    if not campos["vigencia"]:
        vigencia_original = buscar_en_lista(r'(\d{4}\s*[-]?\s*?\d{4})', textos_limpios)
        if vigencia_original:
            campos["vigencia"] = vigencia_original  # 🔄 Usa búsqueda original
    
    # 🧼 14. LIMPIAR FORMATO DE VIGENCIA
    if campos["vigencia"]:
        campos["vigencia"] = re.sub(r'\s+', ' ', campos["vigencia"].replace('-', ' - ').strip())
    
    return campos  # 📦 Retorna todos los campos procesados


# ============================================================
# 🧩 FUNCIÓN AUXILIAR: BUSCAR EN LISTA MEJORADA
# ============================================================
def buscar_en_lista(pattern: str, lista: List[str]) -> str:
    """🔍 Busca un patrón regex en una lista de textos.
    
    🎯 Mejorada con validaciones específicas:
    - 📅 Para fechas: valida que sea fecha plausible
    - 📆 Para vigencias: valida que sean años plausibles
    - 🔍 Para otros: retorna primera coincidencia
    
    Args:
        pattern (str): Patrón regex a buscar
        lista (List[str]): Lista de textos donde buscar
        
    Returns:
        str: Texto encontrado o cadena vacía
    """
    for line in lista:
        # 📅 VALIDACIÓN ESPECIAL PARA FECHAS (DD/MM/YYYY)
        if '\\d{2}/\\d{2}/\\d{4}' in pattern:
            match = re.search(pattern, line)
            if match:
                fecha = match.group(1)
                # ✅ Valida que sea fecha plausible
                try:
                    dia, mes, anio = map(int, fecha.split('/'))
                    # 🕰️ Rango válido: día 1-31, mes 1-12, año 1900-actual
                    if 1 <= dia <= 31 and 1 <= mes <= 12 and 1900 <= anio <= datetime.now().year:
                        return fecha  # ✅ Fecha válida
                except:
                    continue  # 🚫 Error en conversión, sigue buscando
        # 📆 VALIDACIÓN ESPECIAL PARA VIGENCIAS (AAAA-AAAA)
        elif '\\d{4}\\s*[-]' in pattern:
            match = re.search(pattern, line)
            if match:
                vigencia = match.group(1)
                # ✅ Valida que sean años plausibles
                años = re.findall(r'\d{4}', vigencia)
                if len(años) == 2:
                    año1, año2 = int(años[0]), int(años[1])
                    # 🕰️ Rango válido: 1900-2099 y año2 > año1
                    if 1900 <= año1 <= 2099 and 1900 <= año2 <= 2099 and año2 > año1:
                        return vigencia  # ✅ Vigencia válida
        else:
            # 🔍 BÚSQUEDA GENERAL PARA OTROS PATRONES
            match = re.search(pattern, line)
            if match:
                return match.group(1)  # ✅ Coincidencia encontrada
    
    return ""  # 🚫 No se encontró coincidencia


# ============================================================
# 🧩 FUNCIONES AUXILIARES
# ============================================================
def normalizar_textos(texts: List[str]) -> List[str]:
    """🧼 Normaliza una lista de textos OCR.
    
    🎯 Acciones:
    - Elimina espacios múltiples
    - Elimina espacios al inicio/fin
    - Filtra líneas vacías
    
    Args:
        texts (List[str]): Lista de textos crudos
        
    Returns:
        List[str]: Lista de textos normalizados
    """
    limpios: List[str] = []
    for t in texts:
        t2 = re.sub(r'\s+', ' ', (t or '').strip())  # 🧼 Reemplaza múltiples espacios
        if t2:  # ✅ Solo agrega si no está vacío
            limpios.append(t2)
    return limpios


def buscar_seccion(lista: List[str]) -> str:
    """📍 Busca sección electoral en una lista de textos.
    
    🎯 La sección electoral son exactamente 4 dígitos
    
    Args:
        lista (List[str]): Lista de textos donde buscar
        
    Returns:
        str: Sección encontrada o cadena vacía
    """
    for line in lista:
        if re.fullmatch(r'\d{4}', line.strip()):  # 🔢 Exactamente 4 dígitos
            return line.strip()
    return ""  # 🚫 No se encontró sección


# ============================================================
# 🧨 WORKER OCR CON TIMEOUT
# ============================================================
def _ocr_worker(img_bgr: np.ndarray, out_q: mp.Queue) -> None:
    """🏗️ Worker que ejecuta OCR en un proceso separado.
    
    🎯 Propósito: Aislar el OCR en otro proceso para poder
    matarlo si excede el timeout
    
    Args:
        img_bgr (np.ndarray): Imagen en formato BGR (OpenCV)
        out_q (mp.Queue): Cola para devolver resultados
    """
    try:
        engine = _build_ocr_engine()  # 🚀 Crea motor OCR
        result = engine.predict(img_bgr)  # 🔍 Ejecuta OCR
        texts = result[0]["rec_texts"] if result else []  # 📝 Extrae textos
        out_q.put({"ok": True, "texts": texts})  # 📤 Devuelve éxito
    except Exception as e:
        out_q.put({"ok": False, "error": str(e)})  # 📤 Devuelve error


def predict_ocr_texts_with_timeout_kill(img_bgr: np.ndarray, timeout_seconds: int) -> List[str]:
    """⏱️ Ejecuta OCR con timeout y kill de proceso.
    
    🎯 Estrategia:
    1. 🏗️ Crea proceso hijo para OCR
    2. ⏰ Espera timeout_seconds
    3. 💀 Si sigue vivo, lo termina
    4. 📦 Recupera resultados de la cola
    
    Args:
        img_bgr (np.ndarray): Imagen en formato BGR
        timeout_seconds (int): Segundos máximos de espera
        
    Returns:
        List[str]: Lista de textos extraídos
        
    Raises:
        TimeoutError: Si el OCR excede el timeout
        RuntimeError: Si hay error en el OCR
    """
    out_q: mp.Queue = mp.Queue(maxsize=1)  # 📦 Cola para comunicación
    # 🏗️ Crea proceso hijo con el worker OCR
    p = mp.Process(target=_ocr_worker, args=(img_bgr, out_q), daemon=True)
    
    p.start()  # 🚀 Inicia proceso
    p.join(timeout_seconds)  # ⏰ Espera con timeout
    
    # 💀 TERMINAR PROCESO SI SIGUE VIVO (TIMEOUT)
    if p.is_alive():
        try:
            p.terminate()  # 🔴 Termina proceso
        finally:
            p.join(timeout=2)  # ⏳ Espera terminación
        raise TimeoutError("OCR tardó demasiado (proceso terminado)")
    
    # 📦 RECUPERAR RESULTADOS DE LA COLA
    try:
        payload = out_q.get_nowait()  # 📥 Obtiene resultado sin esperar
    except queue.Empty:
        raise RuntimeError("OCR terminó pero no devolvió resultado")
    
    # ❌ MANEJO DE ERRORES DEL WORKER
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "Error desconocido en OCR"))
    
    return payload.get("texts") or []  # ✅ Retorna textos extraídos


# ============================================================
# 🖼️ FUNCIONES DE MANEJO DE IMÁGENES
# ============================================================
def leer_imagen_desde_request(field_name: str = "imagen") -> Optional[np.ndarray]:
    """🖼️ Lee y decodifica una imagen desde un request HTTP multipart.
    
    🎯 Proceso:
    1. 📥 Obtiene archivo del request
    2. 🔢 Lee bytes del archivo
    3. 🖼️ Decodifica a matriz OpenCV
    
    Args:
        field_name (str): Nombre del campo en el formulario (default: "imagen")
        
    Returns:
        Optional[np.ndarray]: Imagen en formato BGR o None si hay error
    """
    if field_name not in request.files:
        return None  # 🚫 No hay archivo en el request
    
    file = request.files[field_name]  # 📂 Obtiene archivo
    data = file.read()  # 🔢 Lee bytes
    if not data:
        return None  # 🚫 Archivo vacío
    
    npimg = np.frombuffer(data, np.uint8)  # 🔢 Convierte bytes a numpy array
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # 🖼️ Decodifica a imagen BGR


# ============================================================
# 🚀 ENDPOINT PRINCIPAL OCR MEJORADO
# ============================================================
@app.route("/ocr", methods=["POST"])
@token_required 
def ocr_anverso_mejorado():
    """
    🪪 ENDPOINT PRINCIPAL: OCR ANVERSO MEJORADO ⭐
    ---
    tags:
      - INE OCR Mejorado
    security:
      - BearerAuth: []  # 🆕 Requiere autenticación
    parameters:
      - name: Authorization
        in: header
        type: string
        required: true
        description: 🔐 Token JWT en formato "Bearer {token}"
      - name: imagen
        in: formData
        type: file
        required: true
        description: 📸 Imagen del anverso de la credencial INE/IFE
    responses:
      200:
        description: ✅ Datos extraídos con validación desde CURP/Clave
      400:
        description: ❌ Falta imagen o imagen inválida
      401:
        description: 🔒 No autorizado - Token inválido o faltante
      408:
        description: ⏱️ OCR tardó demasiado (timeout)
    """
    # 🔍 Obtener información del usuario autenticado (opcional, para logging)
    current_user = getattr(request, 'current_user', {})
    print(f"🔑 Usuario xautenticado: {current_user.get('username', 'Desconocido')}")
    # 🖼️ 1. LEER IMAGEN DEL REQUEST
    img = leer_imagen_desde_request("imagen")
    if img is None:
        return jsonify({"error": "❌ No se envió la imagen o está vacía"}), 400
    
    try:
        # 🔍 2. EJECUTAR OCR CON TIMEOUT
        texts = predict_ocr_texts_with_timeout_kill(img, OCR_TIMEOUT_SECONDS)
    except TimeoutError:
        return jsonify({"error": "❌ La imagen es poco clara"}), 408  # ⏱️ Timeout
    except Exception as e:
        return jsonify({"error": f"❌ Error procesando OCR: {str(e)}"}), 400  # ❌ Error general
    
    # 🪪 3. EXTRAER DATOS CON VALIDACIÓN MEJORADA
    datos = extraer_campos_ine_mejorado(texts)
    
    # 🔧 4. MODO DEBUG (opcional)
    if (request.args.get("debug") or "").strip() in ("1", "true", "True", "yes", "YES"):
        datos["_ocr_texts"] = normalizar_textos(texts)  # 📝 Textos OCR originales
        datos["_tipo_detectado"] = datos.get("tipo_credencial", "DESCONOCIDO")  # 🏷️ Tipo detectado
    
    return jsonify(datos)  # 📦 Retorna datos en JSON


# ============================================================
# 🩺 ENDPOINT HEALTH CHECK
# ============================================================
@app.route("/health", methods=["GET"])
def health_check():
    """🩺 Endpoint para verificar el estado del servicio.
    
    🎯 Uso típico:
    - Monitoreo de salud del servicio
    - Verificación de disponibilidad
    - Balanceadores de carga
    
    Returns:
        JSON con estado del servicio y características
    """
    return jsonify({
        "status": "✅ OK",  # 🟢 Estado del servicio
        "service": "INE OCR API MEJORADO",  # 🏷️ Nombre del servicio
        "version": "2.0.0",  # 🔢 Versión de la API
        "features": ["Clasificación C/D/GH", "Validación CURP/Clave", "Extracción mejorada"]  # ✨ Características
    })


# ============================================================
# 👤🔎 UTILIDADES: SEPARAR NOMBRE CON REGLAS CURP + LIMPIAR COLONIA
# ============================================================

def _solo_letras(s: str) -> str:
    """🔤 Deja solo letras (incluye Ñ/acentos) y espacios."""
    if not s:
        return ""
    s = s.upper().strip()
    s = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _quitar_particulas(tokens: List[str]) -> List[str]:
    """
    🧩 Quita partículas comunes al calcular iniciales CURP (NO para armar el apellido final).
    Ej: DE, DEL, LA, LAS, LOS, Y, MC, MAC, VAN, VON, etc.
    """
    particulas = {
        "DE", "DEL", "LA", "LAS", "LOS", "Y", "MC", "MAC", "VAN", "VON",
        "DA", "DAS", "DO", "DOS", "DI", "DU"
    }
    return [t for t in tokens if t and t not in particulas]


def _primera_vocal_interna(palabra: str) -> str:
    """🔎 Devuelve la primera vocal interna del apellido paterno (para CURP)."""
    if not palabra:
        return ""
    palabra = _solo_letras(palabra).replace(" ", "")
    if len(palabra) < 2:
        return ""
    # vocal interna = desde el 2do char
    m = re.search(r"[AEIOUÁÉÍÓÚÜ]", palabra[1:])
    return m.group(0) if m else ""


def _primer_nombre_para_curp(nombres_tokens: List[str]) -> str:
    """
    👶 Regla común CURP:
    Si el primer nombre es JOSE o MARIA y hay segundo nombre, se usa el segundo.
    """
    if not nombres_tokens:
        return ""
    nt = _quitar_particulas([t.upper() for t in nombres_tokens])
    if not nt:
        return ""
    if nt[0] in {"JOSE", "JOSÉ", "MARIA", "MARÍA"} and len(nt) >= 2:
        return nt[1]
    return nt[0]


def _curp_prefijo_4(ap_pat: str, ap_mat: str, nombres: str) -> str:
    """
    🧬 Construye el prefijo CURP (4) desde partes:
    1) 1ra letra ap_pat
    2) 1ra vocal interna ap_pat
    3) 1ra letra ap_mat
    4) 1ra letra del primer nombre (regla Jose/Maria)
    """
    ap_pat_tokens = _quitar_particulas(_solo_letras(ap_pat).split())
    ap_mat_tokens = _quitar_particulas(_solo_letras(ap_mat).split())
    nom_tokens = _solo_letras(nombres).split()

    ap_pat_base = ap_pat_tokens[0] if ap_pat_tokens else ""
    ap_mat_base = ap_mat_tokens[0] if ap_mat_tokens else ""
    primer_nom = _primer_nombre_para_curp(nom_tokens)

    c1 = ap_pat_base[:1]
    c2 = _primera_vocal_interna(ap_pat_base)
    c3 = ap_mat_base[:1]
    c4 = primer_nom[:1]

    return f"{c1}{c2}{c3}{c4}".upper()


def separar_nombre_por_curp_y_tokens(nombre: str, curp: str) -> Dict[str, str]:
    """
    🧠 Separa 'nombre completo' en:
    - apellido_paterno
    - apellido_materno
    - nombres

    ✅ Estrategia:
    - Tokeniza el nombre
    - Prueba combinaciones (1..3 tokens para ap_pat) + (1..3 tokens para ap_mat)
    - Calcula prefijo CURP(4) y elige la mejor coincidencia vs curp[:4]
    """
    nombre = _solo_letras(nombre)
    curp = (curp or "").upper().strip()

    out = {"apellido_paterno": "", "apellido_materno": "", "nombres": ""}

    tokens = [t for t in nombre.split() if t]
    if len(tokens) < 3:
        # fallback simple
        if len(tokens) == 2:
            out["apellido_paterno"] = tokens[0]
            out["apellido_materno"] = ""
            out["nombres"] = tokens[1]
        elif len(tokens) == 1:
            out["apellido_paterno"] = ""
            out["apellido_materno"] = ""
            out["nombres"] = tokens[0]
        return out

    # Si CURP no viene o está rara, fallback 2 apellidos + resto nombres
    if len(curp) < 4:
        out["apellido_paterno"] = tokens[0]
        out["apellido_materno"] = tokens[1]
        out["nombres"] = " ".join(tokens[2:])
        return out

    objetivo = curp[:4]

    best = None  # (score, ap_pat, ap_mat, nombres)
    # límites razonables para apellidos compuestos
    for i in range(1, min(3, len(tokens) - 1) + 1):        # ap_pat tokens
        for j in range(1, min(3, len(tokens) - i) + 1):    # ap_mat tokens
            if i + j >= len(tokens):
                continue

            ap_pat = " ".join(tokens[:i])
            ap_mat = " ".join(tokens[i:i + j])
            noms = " ".join(tokens[i + j:])

            pref = _curp_prefijo_4(ap_pat, ap_mat, noms)

            # score por coincidencia char a char
            score = sum(1 for a, b in zip(pref, objetivo) if a == b)

            # bonus si coincide todo
            if pref == objetivo:
                score += 10

            # penaliza nombres demasiado cortos
            if len(noms.split()) == 0:
                score -= 5

            cand = (score, ap_pat, ap_mat, noms, pref)
            if best is None or cand[0] > best[0]:
                best = cand

    if best:
        _, ap_pat, ap_mat, noms, _pref = best
        out["apellido_paterno"] = ap_pat
        out["apellido_materno"] = ap_mat
        out["nombres"] = noms
        return out

    # fallback final
    out["apellido_paterno"] = tokens[0]
    out["apellido_materno"] = tokens[1]
    out["nombres"] = " ".join(tokens[2:])
    return out


def limpiar_colonia_con_cp(colonia: str, codigo_postal: str) -> str:
    """
    📮🧹 Si el CP aparece dentro de colonia, lo quita.
    Ej: 'FRACC LA HERRADURA III 77050' + '77050' => 'FRACC LA HERRADURA III'
    """
    colonia = (colonia or "").strip()
    cp = (codigo_postal or "").strip()

    if not colonia or not cp:
        return colonia

    # quita ocurrencias exactas de CP como token (evita romper otros números)
    colonia2 = re.sub(rf"(\b{re.escape(cp)}\b)", "", colonia)
    colonia2 = re.sub(r"\s+", " ", colonia2).strip()

    return colonia2
# ============================================================
# 🧩 ENDPOINT: SEPARAR NOMBRE (CURP + CLAVE ELECTOR) + LIMPIAR COLONIA
# ============================================================
@app.route("/separar-nombre", methods=["POST"])
def api_separar_nombre():
    """
    👤🧬 Separar nombre completo en apellidos y nombres (valida con CURP)
    ---
    tags:
      - Utilidades
    consumes:
      - application/json
    parameters:
      - in: body
        name: payload
        required: true
        schema:
          type: object
          required:
            - nombre
            - curp
            - clave_elector
          properties:
            anio_registro:
              type: string
              example: "2011 02"
            calle:
              type: string
              example: "C LOS MOLINOS 174"
            clave_elector:
              type: string
              example: "CSOLRC93053123H800"
            codigo_postal:
              type: string
              example: "77050"
            colonia:
              type: string
              example: "FRACC LA HERRADURA III 77050"
            curp:
              type: string
              example: "CAOR930531HQRSLC0"
            es_ine:
              type: boolean
              example: true
            estado:
              type: string
              example: "OTHON P. BLANCO, Q. ROO."
            fecha_nacimiento:
              type: string
              example: "31/05/1993"
            nombre:
              type: string
              example: "CASTILLO OLIVERA RICARDO ORLANDO"
            numero:
              type: string
              example: "174"
            pais:
              type: string
              example: "Mex"
            seccion:
              type: string
              example: "0378"
            sexo:
              type: string
              example: "H"
            tipo_credencial:
              type: string
              example: "GH"
            vigencia:
              type: string
              example: "2021 - 2031"
    responses:
      200:
        description: ✅ Objeto original + apellido_paterno, apellido_materno, nombres (y colonia limpia si aplica)
      400:
        description: ❌ Payload inválido o faltan campos requeridos
    """
    data = request.get_json(silent=True) or {}
    nombre = (data.get("nombre") or "").strip()
    curp = (data.get("curp") or "").strip()
    clave_elector = (data.get("clave_elector") or "").strip()

    if not nombre or not curp or not clave_elector:
        return jsonify({
            "error": "❌ Debes enviar al menos: nombre, curp y clave_elector"
        }), 400

    # 🧬 Separación guiada por CURP (y tokens)
    partes = separar_nombre_por_curp_y_tokens(nombre, curp)

    # 📮 Limpieza de colonia quitando CP si viene incrustado
    codigo_postal = (data.get("codigo_postal") or "").strip()
    colonia = (data.get("colonia") or "").strip()
    colonia_limpia = limpiar_colonia_con_cp(colonia, codigo_postal)

    # ✅ Respuesta: mismo objeto + 3 atributos + colonia limpia
    resp = dict(data)
    resp["apellido_paterno"] = partes["apellido_paterno"]
    resp["apellido_materno"] = partes["apellido_materno"]
    resp["nombres"] = partes["nombres"]

    # solo modifica colonia si realmente cambió
    if colonia_limpia and colonia_limpia != colonia:
        resp["colonia"] = colonia_limpia

    return jsonify(resp), 200


# ============================================================
# ▶️ PUNTO DE INICIO DE LA APLICACIÓN
# ============================================================
if __name__ == "__main__":
    # 🚀 Inicia el servidor Flask
    app.run(host="0.0.0.0", port=5001, debug=False)
    # 🌐 host="0.0.0.0": Escucha en todas las interfaces
    # 🔢 port=5001: Puerto del servicio
    # 🐛 debug=False: Modo producción (sin debug)