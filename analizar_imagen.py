"""
analizar_imagen.py
------------------
Carga el modelo ONNX y analiza UNA imagen para determinar
si es o no una flor. Muestra:
  - Estado: "Reconocido ✅" / "No reconocido ❌"
  - Confianza máxima (%)
  - Justificación textual

Uso:
    python analizar_imagen.py ruta/a/imagen.jpg
    python analizar_imagen.py                    ← abre diálogo de selección
"""

import sys, textwrap
import numpy as np
from PIL import Image
import onnxruntime as ort

# ── Umbral de decisión (según pizarra: rango 0.0–1.0) ─────────────
UMBRAL          = 0.5
IMG_SIZE        = 224
MODEL_PATH      = "modelo_flores.onnx"
INPUT_NAME      = "cam_input"
OUTPUT_NAME     = "confidence_score"


def cargar_imagen(ruta: str) -> np.ndarray:
    """Carga y preprocesa la imagen para la CNN."""
    img = Image.open(ruta).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0          # normalizar [0,1]
    return arr[np.newaxis, ...]                            # shape [1,224,224,3]


def justificar(confianza: float, es_flor: bool) -> str:
    """Genera una justificación textual según el nivel de confianza."""
    pct = confianza * 100
    if es_flor:
        if pct >= 90:
            return (f"El modelo detectó con muy alta confianza ({pct:.1f}%) "
                    "características visuales típicas de flores: pétalos, "
                    "colores vivos y texturas orgánicas simétricas.")
        elif pct >= 70:
            return (f"Con confianza moderada-alta ({pct:.1f}%), el modelo "
                    "identificó rasgos compatibles con flores (formas curvas, "
                    "paleta cromática floral), aunque la imagen puede tener "
                    "algo de ruido visual.")
        else:
            return (f"El modelo clasificó la imagen como flor ({pct:.1f}%), "
                    "pero con confianza baja. La imagen podría ser ambigua "
                    "o de baja calidad. Se supera el umbral ({UMBRAL*100:.0f}%).")
    else:
        if pct <= 10:
            return (f"El modelo descartó con altísima certeza ({100-pct:.1f}% "
                    "de confianza de no-flor) que la imagen contenga flores. "
                    "No se detectaron pétalos, formas florales ni colores "
                    "característicos.")
        elif pct <= 30:
            return (f"El modelo no reconoció la imagen como flor "
                    f"(confianza flor: {pct:.1f}%). Las características "
                    "visuales presentes no coinciden con patrones florales.")
        else:
            return (f"La imagen quedó cerca del umbral (confianza flor: "
                    f"{pct:.1f}%), pero no lo superó ({UMBRAL*100:.0f}%). "
                    "Puede haber colores o formas similares a flores, pero "
                    "el modelo las descartó.")


def analizar(ruta_imagen: str):
    # Cargar modelo ONNX
    session = ort.InferenceSession(MODEL_PATH)

    # Preprocesar imagen
    img_array = cargar_imagen(ruta_imagen)

    # Inferencia
    outputs = session.run(
        [OUTPUT_NAME],
        {INPUT_NAME: img_array}
    )
    confianza = float(outputs[0][0][0])   # valor float32 entre 0 y 1

    es_flor   = confianza >= UMBRAL
    estado    = "✅ RECONOCIDO (es una flor)" if es_flor else "❌ NO RECONOCIDO (no es una flor)"
    justif    = justificar(confianza, es_flor)

    # ── Mostrar resultado ─────────────────────────────────────────
    separador = "─" * 55
    print(f"\n{separador}")
    print(f"  Imagen analizada : {ruta_imagen}")
    print(f"  Estado           : {estado}")
    print(f"  Confianza        : {confianza * 100:.2f}%")
    print(f"  Umbral usado     : {UMBRAL * 100:.0f}%")
    print(f"  Justificación    :")
    print(textwrap.fill(f"    {justif}", width=55))
    print(f"{separador}\n")


# ── Punto de entrada ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        # Si no se pasa ruta por argumento, abrir diálogo gráfico
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