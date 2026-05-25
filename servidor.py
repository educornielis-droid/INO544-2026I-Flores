"""
servidor.py
-----------
Servidor web local que conecta la interfaz HTML con el modelo ONNX.

Uso:
    python servidor.py
    
Luego abre en el navegador: http://localhost:5000
"""

import base64, io, json, os
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from PIL import Image
import onnxruntime as ort

# ── Configuración ─────────────────────────────────────────────────
PUERTO      = 5000
IMG_SIZE    = 224
UMBRAL      = 0.5
MODEL_PATH  = "modelo_flores.onnx"
INPUT_NAME  = "cam_input"
OUTPUT_NAME = "confidence_score"
HTML_FILE   = "interfaz_flores.html"

# ── Cargar modelo al iniciar ───────────────────────────────────────
print("🔌 Cargando modelo ONNX...")
session = ort.InferenceSession(MODEL_PATH)
print("✅ Modelo listo")


def preprocesar(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0
    return arr[np.newaxis, ...]


def analizar(img_bytes: bytes) -> dict:
    entrada = preprocesar(img_bytes)
    outputs = session.run([OUTPUT_NAME], {INPUT_NAME: entrada})
    confianza_original = float(outputs[0][0][0])
    # Invertir si el modelo entrenó con etiquetas al revés
    confianza = 1.0 - confianza_original
    es_flor   = confianza >= UMBRAL
    return {
        "es_flor":   es_flor,
        "confianza": round(confianza, 4),
        "umbral":    UMBRAL
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silenciar logs del servidor

    def do_GET(self):
        if self.path in ('/', '/index.html', '/interfaz_flores.html'):
            try:
                with open(HTML_FILE, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_error(404, "Archivo HTML no encontrado")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/analizar':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body   = json.loads(self.rfile.read(length))
                img_b64 = body.get('imagen', '')
                img_bytes = base64.b64decode(img_b64)
                resultado = analizar(img_bytes)
                resp = json.dumps(resultado).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


if __name__ == "__main__":
    server = HTTPServer(('localhost', PUERTO), Handler)
    print(f"\n🌸 FloreScope corriendo en http://localhost:{PUERTO}")
    print("   Abre ese link en tu navegador")
    print("   Presiona Ctrl+C para detener\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Servidor detenido.")