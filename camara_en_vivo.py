"""
camara_en_vivo.py
-----------------
Activa la cámara de la PC y clasifica en tiempo real si lo que
se ve es o no una flor. Muestra el resultado superpuesto sobre
el video con OpenCV.

Controles:
    Q  → salir
    S  → guardar captura de pantalla
"""

import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime

# ── Configuración ─────────────────────────────────────────────────
MODEL_PATH   = "modelo_flores.onnx"
INPUT_NAME   = "cam_input"
OUTPUT_NAME  = "confidence_score"
IMG_SIZE     = 224
UMBRAL       = 0.5
CAMERA_INDEX = 0   # 0 = primera cámara disponible


def preprocesar_frame(frame: np.ndarray) -> np.ndarray:
    """Redimensiona y normaliza un frame BGR de OpenCV."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resz  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    arr   = resz.astype("float32") / 255.0
    return arr[np.newaxis, ...]   # [1, 224, 224, 3]


def dibujar_resultado(frame, confianza: float):
    """Superpone el resultado sobre el frame de video."""
    es_flor = confianza >= UMBRAL
    pct     = confianza * 100

    # Colores: verde = flor | rojo = no flor
    color      = (0, 200, 0) if es_flor else (0, 0, 220)
    etiqueta   = "RECONOCIDO - ES FLOR" if es_flor else "NO RECONOCIDO"
    confianza_texto = f"Confianza: {pct:.1f}%  |  Umbral: {UMBRAL*100:.0f}%"

    h, w = frame.shape[:2]

    # Fondo semitransparente en la parte superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Texto principal
    cv2.putText(frame, etiqueta,
                (15, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Texto de confianza
    cv2.putText(frame, confianza_texto,
                (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    # Barra de confianza
    barra_x   = 15
    barra_y   = 85
    barra_w   = int((w - 30) * confianza)
    cv2.rectangle(frame, (barra_x, barra_y), (w - 15, barra_y + 6), (60, 60, 60), -1)
    cv2.rectangle(frame, (barra_x, barra_y), (barra_x + barra_w, barra_y + 6), color, -1)

    # Instrucciones en la parte inferior
    cv2.putText(frame, "Q: Salir  |  S: Guardar captura",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def main():
    print("🔌 Cargando modelo ONNX...")
    session = ort.InferenceSession(MODEL_PATH)
    print("✅ Modelo cargado")

    print(f"📷 Abriendo cámara (índice {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        print("   Verificá que la cámara esté conectada y no esté en uso por otra app.")
        return

    print("✅ Cámara activa. Mostrando ventana de video...")
    print("   Presioná Q para salir, S para guardar una captura.\n")

    confianza = 0.0   # valor inicial

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  No se pudo leer el frame de la cámara.")
            break

        # Clasificar cada frame
        entrada   = preprocesar_frame(frame)
        outputs   = session.run([OUTPUT_NAME], {INPUT_NAME: entrada})
        confianza = float(outputs[0][0][0])

        # Dibujar resultado sobre el frame
        frame_con_resultado = dibujar_resultado(frame.copy(), confianza)

        cv2.imshow("Detector de Flores — Cámara en Vivo", frame_con_resultado)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q") or tecla == ord("Q"):
            print("👋 Cerrando cámara...")
            break
        elif tecla == ord("s") or tecla == ord("S"):
            nombre = f"captura_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(nombre, frame_con_resultado)
            print(f"📸 Captura guardada como '{nombre}'")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Programa cerrado correctamente.")


if __name__ == "__main__":
    main()