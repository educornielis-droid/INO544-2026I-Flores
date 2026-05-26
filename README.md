# DO YOU TENSORFLOW? — Ves lo que veo 👁️🤖
## IUJO — Feria de Haceres Período I-2026
### Unidad Curricular: INO-544 (Investigación de Operaciones)

---

## 👥 Integrantes y Roles
* **Estudiante 1:** [Eduardo Cornielis] - [30.908.425] - Modelado y Entrenamiento (el que trabajó con el código del modelo)
* **Estudiante 2:** [Cesar Nucete] - [29.548.403] - Dataset y Preprocesamiento (el que buscó y organizó las imágenes)
* **Estudiante 3:** [Andrea Melchor] - [31.170.328] - Pruebas y correciones (probó el modelo y corrigió errores)

---

## 🎯 1. Clase/Tema Seleccionado
* **Tema asignado:** [Flores]
* **Descripción del Objeto:** Las flores se caracterizan visualmente por pétalos de colores vivos y variados (rojos, amarillos, morados, blancos), formas simétricas y radiales, texturas orgánicas suaves, y centros circulares bien definidos. El modelo aprendió a distinguir estas características de objetos sin estas propiedades.
---

## 📊 2. Gestión del Dataset (Ingeniería de Datos)
* **Cantidad de imágenes originales recopiladas:** 8000 (4000 flores + 4000 no flores)
* **Estrategia de Data Augmentation aplicada:** El modelo no aplicó augmentation explícita debiido a que usó las imágenes originales del dataset de TensorFlow (flower_photos) tal como estaban.
    * *Rotación:* [No aplicada] 
    * *Zoom:* [No aplicado]
    * *Cambios de Brillo:* [No aplicados]
    * *Otras transformaciones:* [No aplicadas]
* **Total de imágenes generadas para el entrenamiento:** [8000 (las mismas 4000 originales, sin aumentación)]
* **Resolución y formato estandarizado:** 224x224 píxeles, RGB.

---

## 🧠 3. Arquitectura del Modelo y Entrenamiento
* **Framework utilizado:** [TensorFlow/Keras]
* **Descripción de la Red (CNN):** [Se usó MobileNetV2 como base convolucional preentrenada (congelada), seguida de una capa GlobalAveragePooling2D, una capa densa de 64 neuronas con activación ReLU, una capa Dropout del 30%, y una capa de salida densa de 1 neurona con activación Sigmoid.]. 
* **Hiperparámetros óptimos seleccionados:**
    * *Función de pérdida (Loss):* [Binary Crossentropy] 
    * *Optimizador:* [Adam]
    * *Tasa de Aprendizaje (Learning Rate):* [0.001]
    * *Épocas (Epochs):* [10]
    * *Tamaño de lote (Batch Size):* [4]

### 💡 Justificación Crítica (Control de Autoría)
*Explique detalladamente por qué el equipo eligió esa Tasa de Aprendizaje (Learning Rate) específica y el impacto que tuvo en las gráficas de pérdida durante el laboratorio:* 
> [Escribir aquí la respuesta analítica del equipo. Evite respuestas genéricas generadas por IA]. ------------

---

## 📈 4. Métricas de Rendimiento (Testing - 20%)
* **Precisión final (Accuracy) en la data de test:** [100%]
* **Pérdida final (Loss) en la data de test:** [0.0000]

*(Inserte aquí abajo la captura de pantalla de la gráfica de entrenamiento Accuracy/Loss de su modelo)*
![Gráfica de Entrenamiento](src/grafica_rendimiento.png) ---------------

---

## ⚙️ 5. Especificación de Exportación ONNX
El modelo se ha homologado bajo los estándares requeridos por la interfaz centralizada:
* **Nombre del archivo:** `model/nombre_equipo.onnx`
* **Tensor de Entrada (Input Shape):** `[1, 224, 224, 3]` (Tipo: `float32`)
* **Tensor de Salida (Output Shape):** `[1, 1]` (Tipo: `float32`)
* **Función de activación final:** Sigmoide (Rango de salida de 0.0 a 1.0 para conversión a porcentaje).

---

## 🚀 6. Instrucciones de Ejecución Local
Para replicar el preprocesamiento y el entrenamiento del modelo:

1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/](https://github.com/)[usuario]/[repositorio].git
