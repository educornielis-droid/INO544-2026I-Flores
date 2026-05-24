"""
entrenar_mejorado.py
-------------------
Entrena el modelo con tus 500 flores + todas las categorías negativas.
Optimizado para PCs con poca RAM.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tf2onnx
import pathlib

# ─── Configuración ────────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 8  # Pequeño para no saturar RAM
EPOCHS = 2
MODEL_H5 = "modelo_flores.h5"
MODEL_ONNX = "modelo_flores.onnx"


DATASET_PATH = "C:/Users/Usuario/Desktop/flor"  

# Verificar que la carpeta existe
if not os.path.exists(DATASET_PATH):
    print(f"❌ ERROR: No encuentro la carpeta '{DATASET_PATH}'")
    print("   Cambia la variable DATASET_PATH con la ruta correcta")
    exit(1)

# Verificar que existen las subcarpetas flores/ y no_flores/
if not os.path.exists(os.path.join(DATASET_PATH, "flores")):
    print(f"❌ ERROR: No encuentro la carpeta '{DATASET_PATH}/flores'")
    exit(1)
if not os.path.exists(os.path.join(DATASET_PATH, "no_flores")):
    print(f"❌ ERROR: No encuentro la carpeta '{DATASET_PATH}/no_flores'")
    exit(1)

print("📊 Configuración:")
print(f"   Ruta del dataset: {DATASET_PATH}")
print(f"   Tamaño imágenes: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Épocas: {EPOCHS}")

# ─── 1. Data Augmentation (crea variedad sin más imágenes) ────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% para validación
)

# ─── 2. Cargar imágenes desde carpetas ────────────────────────────
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
    print(f"   {clase} → {indice} (1 = flores, 0 = no flores)")

print(f"\n📊 Total:")
print(f"   Train: {train_generator.samples} imágenes")
print(f"   Validación: {validation_generator.samples} imágenes")

# ─── 3. Modelo CNN (MobileNetV2 pre-entrenada) ────────────────────
print("\n🏗️  Construyendo modelo...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Congelar capas iniciales

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="cam_input")
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence_score")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# ─── 4. Compilar y entrenar ───────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    tf.keras.callbacks.ModelCheckpoint(MODEL_H5, save_best_only=True)
]

print("\n🚀 Comenzando entrenamiento...")
print("   (Esto puede tomar varios minutos u horas dependiendo de tu PC)")
print("   Presiona Ctrl+C para cancelar si es necesario\n")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ─── 5. Evaluar ───────────────────────────────────────────────────
print("\n📊 Evaluación final:")
loss, acc, prec, rec = model.evaluate(validation_generator, verbose=0)
print(f"   ✅ Accuracy: {acc*100:.2f}%")
print(f"   ✅ Precisión: {prec*100:.2f}%")
print(f"   ✅ Recall: {rec*100:.2f}%")

# ─── 6. Exportar a ONNX ───────────────────────────────────────────
print("\n📦 Exportando a ONNX...")

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

print(f"✅ Modelo guardado como '{MODEL_ONNX}'")
print("\n🎉 ¡ENTRENAMIENTO COMPLETO!")
print("   Ya puedes usar 'camara_en_vivo.py' con el nuevo modelo.")