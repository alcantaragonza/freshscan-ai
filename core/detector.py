"""
core/detector.py  –  Captura de cámara, detección de ROI e inferencia en tiempo real.

Flujo:
  1. Hilo de captura lee frames continuamente desde la webcam.
  2. Cada INFERENCE_INTERVAL segundos se llama detect_fruit_roi() sobre el frame,
     y si hay ROI válido se ejecuta FruitPredictor.predict().
  3. Las últimas SMOOTHING_N predicciones se votan por mayoría para estabilizar
     el display y evitar parpadeos.
  4. Si la misma clase prevalece durante CONFIRM_WINDOW segundos continuos (≥2 s),
     se emite on_confirmed() UNA SOLA VEZ (deduplicación por aparición).
  5. Cuando la detección desaparece (ROI None o confianza baja) se reinicia el
     estado de estabilidad, permitiendo que el siguiente objeto sea confirmado.
"""
import time
import threading
import numpy as np
import cv2
from collections import deque, Counter

from core.preprocessor import detect_fruit_roi
from model.predict import FruitPredictor

INFERENCE_INTERVAL = 0.15   # segundos entre inferencias
CONFIRM_WINDOW     = 2.0    # segundos estables antes de guardar en DB
CONFIDENCE_MIN     = 0.60   # umbral mínimo de confianza para considerar detección
SMOOTHING_N        = 5      # tamaño de ventana de suavizado (votos por mayoría)

# Color del bounding box en el video (BGR)
_BOX_COLOR  = (0, 212, 170)
_BOX_THICK  = 2
_FONT       = cv2.FONT_HERSHEY_SIMPLEX


class FruitDetector:
    def __init__(self, camera_index: int = 0,
                 on_result=None, on_confirmed=None):
        self._camera_index  = camera_index
        self._on_result     = on_result       # callback(result_dict, frame)
        self._on_confirmed  = on_confirmed    # callback(result_dict, frame)

        # El predictor se instancia en start() para que errores de modelo
        # sean capturados por el try/except del botón "Iniciar Cámara".
        self._predictor: FruitPredictor | None = None

        self._cap:     cv2.VideoCapture | None = None
        self._thread:  threading.Thread | None = None
        self._running: bool = False

        # Frame compartido entre hilo de captura y hilo principal de UI
        self._frame_lock   = threading.Lock()
        self._latest_frame: np.ndarray | None = None

        # Buffer de suavizado
        self._pred_buffer: deque[str] = deque(maxlen=SMOOTHING_N)

        # Estado de estabilidad / confirmación
        self._stable_label:     str | None  = None
        self._stable_start:     float       = 0.0
        self._already_confirmed: bool       = False

    # ── API pública ───────────────────────────────────────────────────────────

    def start(self):
        """Carga el modelo, abre la cámara y lanza el hilo de captura."""
        if self._running:
            return

        if self._predictor is None:
            self._predictor = FruitPredictor()   # lanza FileNotFoundError si no hay modelo

        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se puede abrir la cámara {self._camera_index}.")

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Detiene el hilo y libera la cámara."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._frame_lock:
            self._latest_frame = None
        self._reset_stability()

    def get_frame(self) -> np.ndarray | None:
        """Devuelve una copia del último frame anotado (thread-safe)."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    # ── Hilo de captura ───────────────────────────────────────────────────────

    def _loop(self):
        last_inference = 0.0

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            now = time.time()

            # Detectar ROI para anotar el frame con el bounding box
            roi, bbox = detect_fruit_roi(frame)

            display = frame.copy()
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), _BOX_COLOR, _BOX_THICK)

            with self._frame_lock:
                self._latest_frame = display

            # Inferencia a ritmo controlado
            if now - last_inference >= INFERENCE_INTERVAL:
                last_inference = now
                if roi is not None:
                    try:
                        result = self._predictor.predict(roi)
                    except Exception:
                        result = None
                else:
                    result = None
                self._process_result(result, frame)

    # ── Lógica de estabilidad y confirmación ──────────────────────────────────

    def _process_result(self, result: dict | None, frame: np.ndarray):
        # Sin detección o confianza insuficiente → emitir "indefinido"
        if result is None or result["confidence"] < CONFIDENCE_MIN:
            self._pred_buffer.clear()
            self._reset_stability()
            if self._on_result:
                self._on_result({"label": "indefinido"}, frame)
            return

        self._pred_buffer.append(result["label"])

        # Necesitamos al menos 2 votos para suavizar
        if len(self._pred_buffer) < 2:
            if self._on_result:
                self._on_result(result, frame)
            return

        # Voto por mayoría sobre la ventana de suavizado
        smoothed_label = Counter(self._pred_buffer).most_common(1)[0][0]
        smoothed       = self._build_smoothed(result, smoothed_label)

        if self._on_result:
            self._on_result(smoothed, frame)

        # Lógica de confirmación con ventana temporal
        now = time.time()
        if smoothed_label == self._stable_label:
            elapsed = now - self._stable_start
            if elapsed >= CONFIRM_WINDOW and not self._already_confirmed:
                self._already_confirmed = True
                if self._on_confirmed:
                    self._on_confirmed(smoothed, frame)
        else:
            # Cambio de clase → reiniciar contador de estabilidad
            self._stable_label      = smoothed_label
            self._stable_start      = now
            self._already_confirmed = False

    def _build_smoothed(self, base_result: dict, smoothed_label: str) -> dict:
        """Construye el resultado con la etiqueta suavizada y sus atributos derivados."""
        if smoothed_label == base_result["label"]:
            return base_result

        parts = smoothed_label.split("-")
        fruit_map = {"banano": "Banano", "manzana": "Manzana"}
        state_map = {
            "maduro":  "maduro", "madura":  "maduro",
            "verde":   "verde",
            "podrido": "podrido", "podrida": "podrido",
        }
        fruit = fruit_map.get(parts[0], parts[0].capitalize())
        state = state_map.get(parts[1], parts[1]) if len(parts) > 1 else "desconocido"

        return {
            "label":      smoothed_label,
            "fruit":      fruit,
            "state":      state,
            "confidence": base_result["confidence"],
            "probs":      base_result.get("probs", []),
        }

    def _reset_stability(self):
        self._stable_label      = None
        self._stable_start      = 0.0
        self._already_confirmed = False
        self._pred_buffer.clear()