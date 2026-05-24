"""
analizar_imagen.py
------------------
Carga el modelo ONNX y analiza UNA imagen para determinar
si es o no una flor. CORREGIDO para que coincida con camara_en_vivo.py

Uso:
    python analizar_imagen.py ruta/a/imagen.jpg
    python analizar_imagen.py                    ← abre diálogo de selección
"""

import sys
import textwrap
import numpy as np
from PIL import Image
import onnxruntime as ort

# ── Configuración ─────────────────────────────────────────────────
IMG_SIZE        = 224
MODEL_PATH      = "modelo_flores.onnx"
INPUT_NAME      = "cam_input"
OUTPUT_NAME     = "confidence_score"


def cargar_imagen(ruta: str) -> np.ndarray:
    """Carga y preprocesa la imagen para la CNN."""
    img = Image.open(ruta).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0
    return arr[np.newaxis, ...]


def justificar(confianza_corregida: float, es_flor: bool) -> str:
    """Genera una justificación textual según el nivel de confianza."""
    pct = confianza_corregida * 100
    if es_flor:
        if pct >= 90:
            return f"El modelo detectó con muy alta confianza ({pct:.1f}%) que la imagen ES una flor."
        elif pct >= 70:
            return f"Con confianza moderada-alta ({pct:.1f}%), la imagen probablemente ES una flor."
        else:
            return f"La imagen fue clasificada como flor ({pct:.1f}%) pero con confianza baja. El umbral es 50%."
    else:
        if pct <= 10:
            return f"El modelo determinó con altísima certeza ({100-pct:.1f}%) que la imagen NO ES una flor."
        elif pct <= 30:
            return f"El modelo NO reconoció la imagen como flor (confianza flor: {pct:.1f}%)."
        else:
            return f"La imagen NO superó el umbral (confianza flor: {pct:.1f}%). El umbral es 50%."


def analizar(ruta_imagen: str):
    print(f"\n📷 Cargando imagen: {ruta_imagen}")
    
    # Cargar modelo ONNX
    session = ort.InferenceSession(MODEL_PATH)

    # Preprocesar imagen
    img_array = cargar_imagen(ruta_imagen)

    # Inferencia
    outputs = session.run([OUTPUT_NAME], {INPUT_NAME: img_array})
    confianza_original = float(outputs[0][0][0])
    
    # ========== MISMA LÓGICA QUE camara_en_vivo.py ==========
    # Invertir la confianza (porque el modelo entrenó al revés)
    confianza_corregida = 1.0 - confianza_original
    es_flor = confianza_corregida >= 0.5
    # ========================================================
    
    justif = justificar(confianza_corregida, es_flor)

    # Mostrar resultado
    separador = "─" * 55
    estado = "✅ RECONOCIDO (es una flor)" if es_flor else "❌ NO RECONOCIDO (no es una flor)"
    
    print(f"{separador}")
    print(f"  Imagen analizada : {ruta_imagen}")
    print(f"  Estado           : {estado}")
    print(f"  Confianza modelo : {confianza_original * 100:.2f}%")
    print(f"  Confianza real   : {confianza_corregida * 100:.2f}%")
    print(f"  Umbral usado     : 50%")
    print(f"  Justificación    :")
    print(textwrap.fill(f"    {justif}", width=55))
    print(f"{separador}\n")


# ── Punto de entrada ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            ruta = filedialog.askopenfilename(
                title="Seleccioná una imagen",
                filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp")]
            )
            if not ruta:
                print("No se seleccionó ninguna imagen.")
                sys.exit(0)
        except Exception:
            print("Uso: python analizar_imagen.py <ruta_imagen.jpg>")
            sys.exit(1)

    analizar(ruta)