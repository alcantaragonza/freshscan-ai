"""
core/preprocessor.py  –  Detección de ROI por color y preprocesamiento de frames.

Estrategia: segmentación HSV multi-rango para aislar el objeto de fruta dominante
en el frame. Retorna la bounding box del contorno más grande que supere el umbral
mínimo de área, junto con el recorte de imagen (ROI) para inferencia.
"""
import cv2
import numpy as np

IMG_SIZE = (128, 128)

# Rangos HSV que cubren los colores predominantes de las 6 clases objetivo.
# Cada entrada es (lower_hsv, upper_hsv).
_HSV_RANGES = [
    # Verde (banano-verde, manzana-verde)
    (np.array([30,  40,  40]), np.array([90, 255, 255])),
    # Amarillo / naranja (banano-maduro)
    (np.array([15,  50,  80]), np.array([30, 255, 255])),
    # Rojo bajo (manzana-madura roja)
    (np.array([ 0,  60,  40]), np.array([10, 255, 255])),
    # Rojo alto (manzana-madura roja, wrap-around HSV)
    (np.array([165, 60,  40]), np.array([180, 255, 255])),
    # Marrón oscuro (frutas podridas)
    (np.array([8,   30,  20]), np.array([22, 160, 120])),
]

# Umbral mínimo de área del contorno (píxeles²) para considerar detección válida.
MIN_AREA = 5_000
# Relleno alrededor del bounding box detectado (píxeles).
_PAD = 24


def detect_fruit_roi(frame: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """
    Detecta la región de interés con la fruta más grande del frame.

    Retorna:
        roi   – recorte BGR de la región detectada, o None si no hay detección.
        bbox  – tupla (x, y, w, h) del bounding box, o None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _HSV_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)

    # Cerrar huecos y eliminar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        return None, None

    x, y, w, h = cv2.boundingRect(largest)
    H, W = frame.shape[:2]
    x1 = max(0, x - _PAD)
    y1 = max(0, y - _PAD)
    x2 = min(W, x + w + _PAD)
    y2 = min(H, y + h + _PAD)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    return roi, (x1, y1, x2 - x1, y2 - y1)