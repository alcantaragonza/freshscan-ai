"""
main.py  –  Punto de entrada del sistema de clasificación de frutas.

Uso:
    python main.py
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # silencia logs de TF en consola

from ui.app import launch

if __name__ == "__main__":
    launch()
