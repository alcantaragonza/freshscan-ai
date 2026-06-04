# Sistema de Clasificación de Frutas en Tiempo Real

Proyecto final del curso **Teoría de Sistemas**. Sistema local en Python que detecta y clasifica **banano** y **manzana** por estado de maduración (**verde / maduro / podrido**) en tiempo real mediante una cámara web USB. Incluye interfaz gráfica, base de datos local y exportación de reportes.

---

## Características

- Detección de región de interés (ROI) dinámica mediante segmentación de color HSV.
- Clasificación con CNN propia entrenada desde cero (sin modelos preentrenados).
- Confirmación con ventana de 2 segundos de estabilidad para evitar registros falsos.
- Persistencia en SQLite con historial de clasificaciones.
- Exportación de reportes en **PDF** y **Excel/CSV**.
- Interfaz gráfica de estilo industrial con Tkinter.

---

## Requisitos del sistema

- Python 3.10 o superior
- Cámara web USB conectada
- Sistema operativo: Windows, macOS o Linux

---

## Instalación

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd proy-final

# Crear y activar entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt
```

> **Linux (Wayland/niri):** si OpenCV no puede abrir ventanas, ejecuta antes:
> ```bash
> export QT_QPA_PLATFORM=xcb
> ```

---

## Uso

### 1. Entrenar el modelo

Debe ejecutarse **una sola vez** antes de usar la aplicación. El dataset debe estar en la carpeta `dataset/` con la siguiente estructura:

```
dataset/
├── banano-maduro/
├── banano-podrido/
├── banano-verde/
├── manzana-madura/
├── manzana-podrida/
└── manzana-verde/
```

```bash
python3 model/train.py
```

El modelo entrenado se guarda automáticamente en `model/saved_model/fruit_classifier.h5`.

### 2. Ejecutar la aplicación

```bash
python3 main.py
```

En la interfaz:
1. Presiona **INICIAR CÁMARA** para comenzar la captura en vivo.
2. Coloca una fruta frente a la cámara sobre un fondo neutro.
3. Tras ~2 segundos de detección estable, el resultado se registra en la base de datos.
4. Usa **Exportar PDF** o **Exportar Excel/CSV** para generar reportes.
5. Presiona **DETENER CÁMARA** para pausar.

---

## Estructura del proyecto

```
proy-final/
├── model/
│   ├── train.py              # Pipeline de entrenamiento CNN
│   ├── predict.py            # Clase FruitPredictor (inferencia)
│   └── saved_model/          # Modelo .h5 y class_names.json (generado)
├── core/
│   ├── detector.py           # FruitDetector: hilo de cámara, ROI, confirmación
│   └── preprocessor.py       # Detección de ROI por segmentación HSV
├── data/
│   ├── db_manager.py         # DBManager: persistencia SQLite
│   ├── db/                   # Base de datos (generado en ejecución)
│   ├── snapshots/            # Capturas de detecciones (generado)
│   └── reports/              # Reportes exportados (generado)
├── ui/
│   ├── app.py                # Interfaz gráfica principal (Tkinter)
│   └── report.py             # Generación de PDF y Excel
├── dataset/                  # Imágenes de entrenamiento (~2384 imágenes)
├── main.py                   # Punto de entrada
├── requirements.txt
└── .gitignore
```

---

## Arquitectura del modelo

- **Tipo:** CNN (Red Neuronal Convolucional) desde cero con TensorFlow/Keras.
- **Entrada:** imágenes 128×128 px, 3 canales RGB.
- **Clases:** 6 (banano-maduro, banano-podrido, banano-verde, manzana-madura, manzana-podrida, manzana-verde).
- **Capas:** 4 bloques Conv2D → BatchNorm → ReLU → MaxPooling → Dropout, seguidos de GlobalAveragePooling y cabeza densa.
- **Augmentación:** RandomFlip, RandomRotation, RandomZoom, RandomContrast (integradas en el modelo, activas solo durante entrenamiento).
- **Optimizador:** Adam con ReduceLROnPlateau y EarlyStopping.

---

## Dependencias principales

| Librería | Uso |
|---|---|
| `tensorflow>=2.13` | Entrenamiento e inferencia CNN |
| `opencv-python>=4.8` | Captura de cámara, procesamiento de imagen |
| `Pillow>=10.0` | Renderizado de frames en Tkinter |
| `numpy>=1.24` | Operaciones matriciales |
| `reportlab>=4.0` | Generación de reportes PDF |
| `openpyxl>=3.1` | Generación de reportes Excel |

> `reportlab` y `openpyxl` son opcionales: si no están instaladas, los reportes se generan en formato TXT y CSV respectivamente.

---

## Notas y limitaciones

- El rendimiento del clasificador depende de las condiciones de iluminación. Se recomienda usar fondo neutro (blanco o negro) y buena iluminación uniforme.
- El dataset incluye ~2384 imágenes con augmentación previa (rotaciones, traslaciones, ruido). Con mayor cantidad de imágenes originales se pueden obtener mejores métricas.
- La inferencia se ejecuta cada ~150 ms para no saturar la CPU en laptops de recursos limitados.
- El sistema no se integra con banda transportadora ni hardware externo; procesa únicamente el video de la cámara web.

---

## Equipo

| Nombre | Carné |
|---|---|
| Brayan Alexander de Leon Pereira | 202308112 |
| Bryan Alexander Perez Santos | 202208024 |
| Andres Fernando Gonzalez Alcantara | 202308061 |

---

*Curso: Teoría de Sistemas — Proyecto Final*