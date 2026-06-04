"""
model/train.py  –  Entrenamiento de CNN desde cero para clasificación de frutas.

Clases (6): banano-maduro, banano-podrido, banano-verde,
            manzana-madura, manzana-podrida, manzana-verde

Uso:
    python model/train.py

El modelo entrenado se guarda en model/saved_model/fruit_classifier.h5
junto con model/saved_model/class_names.json para decodificación en inferencia.
"""
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Hiperparámetros ───────────────────────────────────────────────────────────
DATASET_DIR  = Path("dataset")
MODEL_DIR    = Path("model/saved_model")
IMG_SIZE     = (128, 128)
BATCH_SIZE   = 32
EPOCHS       = 80
VAL_SPLIT    = 0.20
SEED         = 42
LR_INITIAL   = 1e-3


def build_model(num_classes: int) -> keras.Model:
    """
    CNN desde cero:
      - Capa Rescaling interna → recibe imágenes en [0, 255], no requiere normalización externa.
      - Capas de augmentación integradas → activas sólo durante model.fit().
      - 4 bloques Conv2D + BatchNorm + MaxPool + Dropout.
      - GlobalAveragePooling para reducir parámetros vs Flatten.
      - Cabeza densa con Dropout fuerte.
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")

    # Normalización interna: [0,255] → [0,1]
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # Augmentación de datos (sólo durante training)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomContrast(0.12)(x)

    # Bloque 1 – 32 filtros
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.10)(x)

    # Bloque 2 – 64 filtros
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.15)(x)

    # Bloque 3 – 128 filtros
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.20)(x)

    # Bloque 4 – 256 filtros
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Cabeza clasificadora
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name="FruitCNN")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Cargar datasets ───────────────────────────────────────────────────────
    common = dict(
        directory=DATASET_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,
    )

    train_ds = keras.utils.image_dataset_from_directory(
        validation_split=VAL_SPLIT, subset="training", **common
    )
    val_ds = keras.utils.image_dataset_from_directory(
        validation_split=VAL_SPLIT, subset="validation", **common
    )

    class_names  = train_ds.class_names
    num_classes  = len(class_names)
    print(f"\nClases ({num_classes}): {class_names}")

    # Guardar nombres de clase para inferencia
    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)

    # ── Pipeline de datos ─────────────────────────────────────────────────────
    # Cache después de cargar del disco; shuffle/augment se aplican en cada época.
    train_ds = (
        train_ds
        .cache()
        .shuffle(buffer_size=2000, seed=SEED)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = build_model(num_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_INITIAL),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_path = str(MODEL_DIR / "best_model.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=14,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Guardar modelo final (con mejor restaurado por EarlyStopping)
    final_path = str(MODEL_DIR / "fruit_classifier.h5")
    model.save(final_path)

    best_val = max(history.history["val_accuracy"])
    print(f"\n✓ Modelo guardado en: {final_path}")
    print(f"  Mejor val_accuracy: {best_val:.4f} ({best_val*100:.1f}%)")


if __name__ == "__main__":
    # Reduce verbosidad de TF en consola
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()