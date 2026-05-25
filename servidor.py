"""
servidor.py - usa onnxruntime 1.10.0 compatible con Windows 7
"""

import base64, io, json
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from PIL import Image
import onnxruntime as ort
import warnings
warnings.filterwarnings("ignore")

PUERTO     = 5000
IMG_SIZE   = 224
UMBRAL     = 0.5
MODEL_PATH = "modelo_flores.onnx"
HTML_FILE  = "interfaz_flores.html"
INPUT_NAME  = "cam_input"
OUTPUT_NAME = "confidence_score"

print("🔌 Cargando modelo ONNX...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
print("✅ Modelo listo")


def preprocesar(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0
    return arr[np.newaxis, ...]


def analizar(img_bytes):
    entrada = preprocesar(img_bytes)
    outputs = session.run([OUTPUT_NAME], {INPUT_NAME: entrada})
    conf_original = float(outputs[0][0][0])
    confianza = 1.0 - conf_original
    es_flor   = confianza >= UMBRAL
    print(f"✅ {'FLOR' if es_flor else 'NO FLOR'} | Confianza: {confianza*100:.1f}%")
    return {"es_flor": es_flor, "confianza": round(confianza, 4), "umbral": UMBRAL}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, f, *a): pass

    def send_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ('/', '/interfaz_flores.html'):
            try:
                with open(HTML_FILE, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_cors()
                self.end_headers()
                self.wfile.write(content)
            except:
                self.send_error(404)
        else:
            self.send_response(200)
            self.end_headers()

    def do_POST(self):
        if self.path == '/analizar':
            try:
                length = int(self.headers.get('Content-Length', 0))
                raw = b''
                while len(raw) < length:
                    chunk = self.rfile.read(min(65536, length - len(raw)))
                    if not chunk: break
                    raw += chunk

                body    = json.loads(raw.decode('utf-8'))
                img_b64 = body.get('imagen', '')
                if ',' in img_b64:
                    img_b64 = img_b64.split(',')[1]

                resultado = analizar(base64.b64decode(img_b64))
                resp = json.dumps(resultado).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(resp)))
                self.send_cors()
                self.end_headers()
                self.wfile.write(resp)
                self.wfile.flush()
            except Exception as e:
                print(f"❌ Error: {e}")
                resp = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_cors()
                self.end_headers()
                self.wfile.write(resp)
        else:
            self.send_error(404)


if __name__ == "__main__":
    server = HTTPServer(('localhost', PUERTO), Handler)
    print(f"\n🌸 FloreScope en http://localhost:{PUERTO}")
    print("   Abre ese link en tu navegador")
    print("   Ctrl+C para detener\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Servidor detenido.")