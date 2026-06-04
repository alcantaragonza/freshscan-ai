"""
model/predict.py  –  Carga del modelo entrenado y función de inferencia.

El modelo incluye internamente la capa Rescaling(1/255), por lo que
predict() acepta imágenes BGR uint8 o float32 en rango [0, 255].
"""
import json
import numpy as np
import cv2
from pathlib import Path

MODEL_DIR  = Path("model/saved_model")
IMG_SIZE   = (128, 128)

# Mapeos para parsear nombres de carpeta → (fruta_display, estado_normalizado)
_FRUIT_MAP = {"banano": "Banano", "manzana": "Manzana"}
_STATE_MAP = {
    "maduro":  "maduro",
    "madura":  "maduro",
    "verde":   "verde",
    "podrido": "podrido",
    "podrida": "podrido",
}


def _parse_class(name: str) -> tuple[str, str]:
    """'manzana-madura' → ('Manzana', 'maduro')"""
    parts = name.split("-")
    fruit = _FRUIT_MAP.get(parts[0], parts[0].capitalize())
    state = _STATE_MAP.get(parts[1], parts[1]) if len(parts) > 1 else "desconocido"
    return fruit, state


class FruitPredictor:
    def __init__(self):
        import tensorflow as tf
        model_path = MODEL_DIR / "fruit_classifier.h5"
        names_path = MODEL_DIR / "class_names.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en '{model_path}'.\n"
                "Ejecuta primero:  python model/train.py"
            )

        self._model = tf.keras.models.load_model(str(model_path))

        with open(names_path, encoding="utf-8") as f:
            self.class_names: list[str] = json.load(f)

    def predict(self, image: np.ndarray) -> dict:
        """
        Clasifica un frame o ROI BGR.

        Retorna dict con:
            label      – nombre de clase (p.ej. 'manzana-madura')
            fruit      – 'Banano' | 'Manzana'
            state      – 'verde' | 'maduro' | 'podrido'
            confidence – probabilidad de la clase ganadora [0, 1]
            probs      – vector completo de probabilidades
        """
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE)
        batch   = np.expand_dims(resized.astype("float32"), 0)   # [0-255]

        probs      = self._model.predict(batch, verbose=0)[0]
        idx        = int(np.argmax(probs))
        confidence = float(probs[idx])
        label      = self.class_names[idx]
        fruit, state = _parse_class(label)

        return {
            "label":      label,
            "fruit":      fruit,
            "state":      state,
            "confidence": confidence,
            "probs":      probs.tolist(),
        }