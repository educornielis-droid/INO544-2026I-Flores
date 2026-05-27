"""
camara_en_vivo.py
-----------------
Activa la cámara de la PC y clasifica en tiempo real si lo que
se ve es o no una flor. Muestra el resultado superpuesto sobre
el video con OpenCV.

VERSIÓN CON INVERSIÓN - El modelo aprendió al revés (flores dan confianza baja)

Controles:
    Q  → salir
    S  → guardar captura de pantalla
"""

import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime

# ── Configuración ─────────────────────────────────────────────────
MODEL_PATH   = "model/INO544-2026I-Flores.onnx"
INPUT_NAME   = "cam_input"
OUTPUT_NAME  = "confidence_score"
IMG_SIZE     = 224
CAMERA_INDEX = 0
UMBRAL       = 0.5          # 50% de confianza para decidir


def preprocesar_frame(frame: np.ndarray) -> np.ndarray:
    """Redimensiona y normaliza un frame BGR de OpenCV."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resz  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    arr   = resz.astype("float32") / 255.0
    return arr[np.newaxis, ...]   # [1, 224, 224, 3]


def dibujar_resultado(frame, confianza_original: float):
    """
    Superpone el resultado sobre el frame de video.
    INVERTIR la confianza (modelo aprendió al revés: flores dan confianza baja)
    """
    # INVERTIR confianza
    confianza = 1.0 - confianza_original
    es_flor = confianza >= UMBRAL
    pct = confianza * 100

    # Colores: verde = flor | rojo = no flor
    color = (0, 200, 0) if es_flor else (0, 0, 220)
    etiqueta = "🌼 ES FLOR" if es_flor else "❌ NO ES FLOR"
    confianza_texto = f"Confianza: {pct:.1f}%  |  Umbral: {UMBRAL*100:.0f}%"

    h, w = frame.shape[:2]

    # Fondo semitransparente en la parte superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Texto principal
    cv2.putText(frame, etiqueta,
                (15, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Texto de confianza
    cv2.putText(frame, confianza_texto,
                (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # Barra de confianza
    barra_x = 15
    barra_y = 80
    barra_w = int((w - 30) * confianza)
    cv2.rectangle(frame, (barra_x, barra_y), (w - 15, barra_y + 8), (60, 60, 60), -1)
    cv2.rectangle(frame, (barra_x, barra_y), (barra_x + barra_w, barra_y + 8), color, -1)

    # Instrucciones en la parte inferior
    cv2.putText(frame, "Q: Salir  |  S: Guardar captura",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def main():
    print("=" * 60)
    print("🔌 Cargando modelo ONNX...")
    print("   (esto puede tardar unos segundos)")
    
    try:
        session = ort.InferenceSession(MODEL_PATH)
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        print(f"   Verifica que '{MODEL_PATH}' existe en la carpeta")
        return

    print(f"\n📷 Abriendo cámara (índice {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        print("   Verificá que la cámara esté conectada y no esté en uso por otra app.")
        return

    print("✅ Cámara activa")
    print("\n" + "=" * 60)
    print("🎯 Detector de Flores - Con inversión")
    print(f"   • Umbral de decisión: {UMBRAL*100:.0f}%")
    print("   • Inversión activada (modelo aprendió al revés)")
    print("   • Apunta a una flor para probar")
    print("\n⌨️  Controles:")
    print("   Q → Salir")
    print("   S → Guardar captura")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  No se pudo leer el frame de la cámara.")
            break

        # Clasificar cada frame
        entrada = preprocesar_frame(frame)
        outputs = session.run([OUTPUT_NAME], {INPUT_NAME: entrada})
        confianza_original = float(outputs[0][0][0])
        
        # Dibujar resultado (con inversión)
        frame_con_resultado = dibujar_resultado(frame.copy(), confianza_original)

        cv2.imshow("Detector de Flores", frame_con_resultado)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q") or tecla == ord("Q"):
            print("👋 Cerrando cámara...")
            break
        elif tecla == ord("s") or tecla == ord("S"):
            nombre = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(nombre, frame_con_resultado)
            print(f"📸 Captura guardada como '{nombre}'")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Programa cerrado correctamente.")


if __name__ == "__main__":
    main()