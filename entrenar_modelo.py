"""
entrenar_modelo.py
-------------------
Entrena el modelo con tus 4000 flores + 3969 categorías negativas.
Dataset balanceado (50% flores, 50% no flores).
Optimizado para PCs con poca RAM.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import tf2onnx
import matplotlib.pyplot as plt  # ← NUEVO: para gráficas
import pickle  # ← NUEVO: para guardar history

# ─── Configuración optimizada ─────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32          # Óptimo para 8000 imágenes
EPOCHS = 10              # Early stopping detendrá antes si es necesario
MODEL_H5 = "modelo_flores.h5"
MODEL_ONNX = "modelo_flores.onnx"

#  VERIFICA QUE ESTA RUTA SEA CORRECTA 
DATASET_PATH = "C:\\Users\\Usuario\\Desktop\\flor" 

# ─── Validación de rutas ─────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    print(f"❌ ERROR: No encuentro la carpeta '{DATASET_PATH}'")
    print("   Cambia la variable DATASET_PATH con la ruta correcta")
    exit(1)

if not os.path.exists(os.path.join(DATASET_PATH, "flores")):
    print(f" ERROR: No encuentro la carpeta '{DATASET_PATH}/flores'")
    exit(1)
if not os.path.exists(os.path.join(DATASET_PATH, "no_flores")):
    print(f" ERROR: No encuentro la carpeta '{DATASET_PATH}/no_flores'")
    exit(1)

print("="*70)
print("🌸 ENTRENAMIENTO DE DETECTOR DE FLORES - DATASET BALANCEADO 🌸")
print("="*70)
print("📊 Configuración:")
print(f"   Ruta del dataset: {DATASET_PATH}")
print(f"   Tamaño imágenes: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Épocas: {EPOCHS}")
print("="*70)

# ─── 1. Data Augmentation (variedad controlada) ───────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,          # Suficiente para flores
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2         # 20% para validación
)

# ─── 2. Cargar imágenes ───────────────────────────────────────────
print("\n📂 Cargando imágenes...")

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

print(f"\n✅ Clases encontradas:")
for clase, indice in train_generator.class_indices.items():
    print(f"   📁 {clase} → {indice} (1 = flores, 0 = no flores)")

print(f"\n📊 Total de imágenes:")
print(f"   🚂 Entrenamiento: {train_generator.samples} imágenes")
print(f"   🧪 Validación: {validation_generator.samples} imágenes")

# ─── 3. Modelo CNN con regularización suave ───────────────────────
print("\n🏗️  Construyendo modelo MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Congelar capas pre-entrenadas

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="cam_input")
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence_score")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

print(f"\n📊 Total parámetros entrenables: {model.count_params():,}")

# ─── 4. Compilar ──────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# ─── 5. Callbacks (evitan sobreajuste) ───────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_H5,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ─── 6. Entrenar ──────────────────────────────────────────────────
print("\n" + "="*70)
print("🚀 COMIENZA EL ENTRENAMIENTO...")
print(f"   Total imágenes: {train_generator.samples}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Pasos por época: {train_generator.samples // BATCH_SIZE}")
print("   Presiona Ctrl+C para cancelar")
print("="*70 + "\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ─── 7. Guardar history (para graficar después) ───────────────────
print("\n💾 Guardando historial de entrenamiento...")
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("   ✅ Historial guardado como 'history.pkl'")

# ─── 8. Evaluación final ──────────────────────────────────────────
print("\n" + "="*70)
print("📊 EVALUACIÓN FINAL")
print("="*70)

loss, acc, prec, rec = model.evaluate(validation_generator, verbose=0)
print(f"   ✅ Accuracy:  {acc*100:.2f}%")
print(f"   ✅ Precisión: {prec*100:.2f}%")
print(f"   ✅ Recall:    {rec*100:.2f}%")

# Verificar sobreajuste
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"\n📈 Comparación:")
print(f"   Accuracy entrenamiento: {train_acc*100:.2f}%")
print(f"   Accuracy validación:    {val_acc*100:.2f}%")

if train_acc - val_acc > 0.15:
    print(f"   ⚠️ Posible sobreajuste (diferencia > 15%)")
else:
    print(f"   ✅ Modelo bien generalizado")

# ─── 9. Generar gráficas ──────────────────────────────────────────
print("\n📊 Generando gráficas de entrenamiento...")

plt.figure(figsize=(14, 5))

# Gráfica 1: Pérdida (Loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-o', label='Entrenamiento', linewidth=2, markersize=4)
plt.plot(history.history['val_loss'], 'r-s', label='Validación', linewidth=2, markersize=4)
plt.title('Pérdida (Loss) durante el entrenamiento', fontsize=12, fontweight='bold')
plt.xlabel('Épocas', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(len(history.history['loss'])))

# Gráfica 2: Precisión (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-o', label='Entrenamiento', linewidth=2, markersize=4)
plt.plot(history.history['val_accuracy'], 'r-s', label='Validación', linewidth=2, markersize=4)
plt.title('Precisión (Accuracy) durante el entrenamiento', fontsize=12, fontweight='bold')
plt.xlabel('Épocas', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(len(history.history['accuracy'])))
plt.ylim([0, 1])

plt.tight_layout()
plt.savefig('graficas_entrenamiento.png', dpi=150, bbox_inches='tight')
print("   ✅ Gráficas guardadas como 'graficas_entrenamiento.png'")
plt.show()

# ─── 10. Exportar a ONNX ──────────────────────────────────────────
print("\n📦 Exportando a ONNX...")
print("   (esto puede tomar unos segundos)")

# Cargar los mejores pesos
model.load_weights(MODEL_H5)

input_signature = [
    tf.TensorSpec(shape=[1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32, name="cam_input")
]

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=12,
    output_path=MODEL_ONNX
)

# ─── 11. Recomendaciones finales ──────────────────────────────────
print("\n" + "="*70)
print("🎉 ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
print("="*70)
print(f"✅ Modelo guardado como: {MODEL_ONNX}")
print(f"✅ Pesos guardados como: {MODEL_H5}")
print(f"✅ Historial guardado como: history.pkl")
print(f"✅ Gráficas guardadas como: graficas_entrenamiento.png")
print(f"✅ Dataset usado: {train_generator.samples + validation_generator.samples} imágenes")
print(f"   - Flores: ~{train_generator.samples // 2 + validation_generator.samples // 2}")
print(f"   - No flores: ~{train_generator.samples // 2 + validation_generator.samples // 2}")
print("\n📌 PRÓXIMOS PASOS:")
print("   1. Probar con cámara:")
print("      python camara_en_vivo.py")
print("")
print("   2. Probar servidor web:")
print("      python servidor.py")
print("      http://localhost:5000")
print("")
print("   3. Probar imagen específica:")
print("      python analizar_imagen.py ruta/de/tu/imagen.jpg")
print("")
print("   4. Ver gráficas:")
print("      Abre 'graficas_entrenamiento.png'")
print("="*70)