"""
entrenar_modelo.py
------------------
Versión ultra-liviana para PCs con muy poca RAM (Windows 7).
Usa solo 200 imágenes de cada clase para evitar MemoryError.
"""

import os, pathlib, numpy as np, tensorflow as tf, tf2onnx
from PIL import Image

# ─── Configuración ────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 4
EPOCHS      = 10
MODEL_H5    = "modelo_flores.h5"
MODEL_ONNX  = "modelo_flores.onnx"
MAX_FLORES  = 200
MAX_NEG     = 200

# ─── 1. Dataset de flores ─────────────────────────────────────────
print("📥 Descargando dataset de flores...")
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(
    "flower_photos", origin=dataset_url, untar=True
)
data_dir = pathlib.Path(data_dir)

# Cargar flores de a 1 imagen para no saturar RAM
print("🌸 Cargando flores de a una...")
flower_images = []
count = 0
for img_path in data_dir.glob("*/*.jpg"):
    if count >= MAX_FLORES:
        break
    try:
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype="float32") / 255.0
        flower_images.append(arr)
        count += 1
    except:
        continue
    if count % 50 == 0:
        print(f"   {count}/{MAX_FLORES} flores...")

flower_images = np.array(flower_images, dtype="float32")
print(f"✅ Flores: {len(flower_images)}")

# ─── 2. Negativos con CIFAR-10 ────────────────────────────────────
print("📥 Descargando negativos (CIFAR-10)...")
(x_cifar, y_cifar), _ = tf.keras.datasets.cifar10.load_data()
non_flower_classes = [0, 1, 3, 5, 6, 8, 9]
mask = np.isin(y_cifar.flatten(), non_flower_classes)
x_neg_raw = x_cifar[mask][:MAX_NEG]

print("🔄 Redimensionando negativos de a 10...")
x_neg_list = []
for i in range(0, len(x_neg_raw), 10):
    batch = x_neg_raw[i:i+10]
    resized = tf.image.resize(batch, [IMG_SIZE, IMG_SIZE]).numpy().astype("float32") / 255.0
    x_neg_list.append(resized)
x_neg = np.concatenate(x_neg_list, axis=0)
print(f"✅ Negativos: {len(x_neg)}")

# Liberar lo que no se necesita más
del x_cifar, y_cifar, x_neg_raw, x_neg_list

# ─── 3. Combinar ──────────────────────────────────────────────────
X = np.concatenate([flower_images, x_neg], axis=0)
y = np.concatenate([
    np.ones(len(flower_images),  dtype="float32"),
    np.zeros(len(x_neg),         dtype="float32")
])
del flower_images, x_neg

idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
del X
print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ─── 4. Modelo CNN ────────────────────────────────────────────────
print("🏗️  Construyendo modelo...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="cam_input")
x       = base_model(inputs, training=False)
x       = tf.keras.layers.GlobalAveragePooling2D()(x)
x       = tf.keras.layers.Dense(64, activation="relu")(x)
x       = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence_score")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# ─── 5. Entrenar ──────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_H5, save_best_only=True)
]

print("🚀 Entrenando...")
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ─── 6. Evaluar ───────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Resultado → Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

# ─── 7. Exportar a ONNX ───────────────────────────────────────────
print("📦 Exportando a ONNX...")
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
print("🎉 ¡Entrenamiento completo!")