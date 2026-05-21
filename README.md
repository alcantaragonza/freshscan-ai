# freshscan-ai

Proyecto final – Teoría de Sistemas  
Sistema local de visión por computadora para clasificar **banano** y **manzana**  
en estado **verde**, **maduro** o **podrido**, usando una CNN propia con TensorFlow/Keras.

---

## Requisitos

- Python 3.9–3.11
- Webcam USB conectada
- ~2 GB RAM disponible

## Instalación

```bash
git clone https://github.com/alcantaragonza/freshscan-ai.git
cd clasificador-frutas
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

## Estructura del dataset

Organiza las imágenes en:
```
data/dataset/
├── banano/verde/       # ~100-150 imágenes .jpg/.png
├── banano/maduro/
├── banano/podrido/
├── manzana/verde/
├── manzana/maduro/
└── manzana/podrido/
```

## Entrenamiento del modelo

```bash
python model/train.py
```
El modelo se guarda en `model/saved_model/best_model.h5`.  
Ajusta `EPOCHS`, `BATCH_SIZE` e `IMG_SIZE` en `train.py` según tus recursos.

## Ejecución del sistema

```bash
python main.py
```

1. Haz clic en **▶ INICIAR CÁMARA**.
2. Apunta la webcam a una fruta dentro del recuadro ROI verde.
3. El sistema mostrará tipo de fruta, estado y confianza en tiempo real.
4. Cada clasificación se registra automáticamente en la base de datos SQLite.
5. Usa los botones de exportación para generar reportes PDF y Excel.

## Dependencias opcionales

| Librería | Uso | Fallback |
|----------|-----|----------|
| `reportlab` | Reporte PDF | .txt plano |
| `openpyxl` | Reporte Excel .xlsx | .csv |

## Notas técnicas

- CNN propia de 4 bloques convolucionales (sin transfer learning).
- Data augmentation: flip, rotación ±10°, zoom ±10%, brillo ±15%.
- Suavizado de predicción: ventana de las últimas 5 inferencias.
- Registro en SQLite evita duplicados: mínimo 2 s entre registros iguales.
- Inferencia cada 0.5 s para no saturar CPU.

## Limitaciones

- Con 100-150 imágenes por clase el modelo puede presentar overfitting; el early stopping y dropout lo mitigan.
- Requiere fondo relativamente limpio y buena iluminación para mayor precisión.
- El modelo debe ser reentrenado si se agregan nuevas clases.

## Mejoras futuras

- Aumentar dataset con más imágenes.
- Implementar detección por bounding box real (ROI dinámico).
- Exportar gráficas de accuracy/loss al PDF.
- Agregar autenticación de usuario en la GUI.
