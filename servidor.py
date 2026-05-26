"""
servidor.py - Servidor web para detector de flores

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
    confianza_original = float(outputs[0][0][0])
    confianza = 1.0 - confianza_original  # Inversión (modelo aprendió al revés)
    es_flor = confianza >= UMBRAL
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
        if self.path == '/' or self.path == '/interfaz_flores.html':
            try:
                with open(HTML_FILE, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_cors()
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self.send_error(404, f"Archivo {HTML_FILE} no encontrado")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/analizar':
            try:
                length = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode('utf-8'))
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
            except ConnectionAbortedError:
                pass
            except BrokenPipeError:
                pass
            except Exception as e:
                print(f"❌ Error: {e}")
                try:
                    resp = json.dumps({"error": str(e)}).encode()
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.send_cors()
                    self.end_headers()
                    self.wfile.write(resp)
                except:
                    pass
        else:
            self.send_error(404)

if __name__ == "__main__":
    server = HTTPServer(('localhost', PUERTO), Handler)
    print(f"\n🌸 FloreScope - Servidor activo")
    print(f"   🌐 http://localhost:{PUERTO}")
    print("   • Interfaz con diseño original")
    print("   • Cámara en vivo integrada")
    print("   • Análisis de imágenes por archivo")
    print("   • Presiona Ctrl+C para detener\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Servidor detenido.")
        server.shutdown()