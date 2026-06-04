"""
ui/app.py  –  Interfaz gráfica principal (Tkinter). Estilo consola industrial.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import datetime
import os
import sqlite3
from PIL import Image, ImageTk
import cv2
import numpy as np

from core.detector import FruitDetector
from ui.report import generate_pdf, generate_excel
from data.db_manager import DBManager

# ── Paleta de colores industrial ──────────────────────────────────────────────
BG_DARK   = "#1a1a2e"
BG_MID    = "#16213e"
BG_PANEL  = "#0f3460"
ACCENT    = "#00d4aa"
WARN      = "#e94560"
TEXT_MAIN = "#e0e0e0"
TEXT_DIM  = "#8a8a9a"
FONT_MONO = ("Courier New", 10)
FONT_LBL  = ("Segoe UI", 10)
FONT_BIG  = ("Segoe UI", 18, "bold")
FONT_MED  = ("Segoe UI", 12, "bold")

STATE_COLORS = {
    "verde":   "#2ecc71",
    "maduro":  "#f39c12",
    "podrido": "#e74c3c",
    "":        TEXT_DIM,
    "indefinido": TEXT_DIM,
}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clasificador de Frutas · Visión en Tiempo Real")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.geometry("1100x700")
        self.minsize(900, 600)

        self.db       = DBManager()
        self.detector = FruitDetector(camera_index=0,
                                      on_result=self._on_result,
                                      on_confirmed=self._on_confirmed)
        self._result_lock    = threading.Lock()
        self._pending_result = None

        self._build_ui()
        self._update_video_loop()
        self._update_result_loop()

    # ── Construcción de UI ────────────────────────────────────────────────────
    def _build_ui(self):
        # ---- Barra superior ----
        top_bar = tk.Frame(self, bg=BG_PANEL, height=48)
        top_bar.pack(fill="x", side="top")
        tk.Label(top_bar, text="🍌🍎  SISTEMA DE CLASIFICACIÓN DE FRUTAS",
                 bg=BG_PANEL, fg=ACCENT, font=("Segoe UI", 13, "bold"),
                 padx=16).pack(side="left", pady=8)
        self._lbl_status = tk.Label(top_bar, text="● DETENIDO",
                                    bg=BG_PANEL, fg=WARN,
                                    font=("Segoe UI", 10, "bold"), padx=10)
        self._lbl_status.pack(side="right", pady=8, padx=8)

        # ---- Contenedor principal ----
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=10, pady=6)

        # ---- Panel izquierdo: video ----
        left = tk.Frame(main, bg=BG_MID, bd=1, relief="flat")
        left.pack(side="left", fill="both", expand=True, padx=(0,6))

        tk.Label(left, text="FEED DE CÁMARA", bg=BG_MID, fg=TEXT_DIM,
                 font=FONT_LBL).pack(anchor="w", padx=8, pady=(6,0))
        self._canvas = tk.Canvas(left, bg="#000", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=6, pady=6)

        # ---- Panel derecho ----
        right = tk.Frame(main, bg=BG_DARK, width=320)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # · Resultado actual ·
        res_frame = tk.LabelFrame(right, text=" CLASIFICACIÓN ACTUAL ",
                                  bg=BG_DARK, fg=ACCENT, font=FONT_LBL,
                                  bd=1, relief="groove")
        res_frame.pack(fill="x", padx=4, pady=(0,6))

        self._lbl_fruit = tk.Label(res_frame, text="—", bg=BG_DARK,
                                   fg=TEXT_MAIN, font=FONT_BIG)
        self._lbl_fruit.pack(pady=(8,2))
        self._lbl_state = tk.Label(res_frame, text="—", bg=BG_DARK,
                                   fg=TEXT_DIM, font=("Segoe UI", 14, "bold"))
        self._lbl_state.pack(pady=(0,4))
        self._lbl_conf  = tk.Label(res_frame, text="Confianza: —", bg=BG_DARK,
                                   fg=TEXT_DIM, font=FONT_LBL)
        self._lbl_conf.pack(pady=(0,8))

        # · Controles ·
        ctrl_frame = tk.Frame(right, bg=BG_DARK)
        ctrl_frame.pack(fill="x", padx=4, pady=4)

        self._btn_start = tk.Button(ctrl_frame, text="▶  INICIAR CÁMARA",
                                    bg=ACCENT, fg="#000", font=FONT_LBL,
                                    activebackground="#00b894",
                                    relief="flat", cursor="hand2",
                                    command=self._start_camera)
        self._btn_start.pack(fill="x", pady=(0,4))

        self._btn_stop = tk.Button(ctrl_frame, text="■  DETENER CÁMARA",
                                   bg=WARN, fg="#fff", font=FONT_LBL,
                                   activebackground="#c0392b",
                                   relief="flat", cursor="hand2",
                                   state="disabled",
                                   command=self._stop_camera)
        self._btn_stop.pack(fill="x", pady=(0,8))

        btn_pdf = tk.Button(ctrl_frame, text="📄  Exportar PDF",
                            bg=BG_PANEL, fg=TEXT_MAIN, font=FONT_LBL,
                            relief="flat", cursor="hand2",
                            command=self._export_pdf)
        btn_pdf.pack(fill="x", pady=2)

        btn_xls = tk.Button(ctrl_frame, text="📊  Exportar Excel/CSV",
                            bg=BG_PANEL, fg=TEXT_MAIN, font=FONT_LBL,
                            relief="flat", cursor="hand2",
                            command=self._export_excel)
        btn_xls.pack(fill="x", pady=2)

        # · Historial reciente ·
        hist_frame = tk.LabelFrame(right, text=" HISTORIAL RECIENTE ",
                                   bg=BG_DARK, fg=ACCENT, font=FONT_LBL,
                                   bd=1, relief="groove")
        hist_frame.pack(fill="both", expand=True, padx=4, pady=(8,0))

        self._hist_text = tk.Text(hist_frame, bg=BG_MID, fg=TEXT_MAIN,
                                  font=FONT_MONO, state="disabled",
                                  wrap="none", height=10)
        scr = ttk.Scrollbar(hist_frame, command=self._hist_text.yview)
        self._hist_text.configure(yscrollcommand=scr.set)
        scr.pack(side="right", fill="y")
        self._hist_text.pack(fill="both", expand=True, padx=4, pady=4)

        self._refresh_history()

    # ── Cámara ────────────────────────────────────────────────────────────────
    def _start_camera(self):
        try:
            self.detector.start()
            self._btn_start.config(state="disabled")
            self._btn_stop.config(state="normal")
            self._lbl_status.config(text="● EN VIVO", fg=ACCENT)
        except Exception as e:
            messagebox.showerror("Error de cámara", str(e))

    def _stop_camera(self):
        self.detector.stop()
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._lbl_status.config(text="● DETENIDO", fg=WARN)
        self._lbl_fruit.config(text="—", fg=TEXT_MAIN)
        self._lbl_state.config(text="—", fg=TEXT_DIM)
        self._lbl_conf.config(text="Confianza: —")

    # ── Callbacks del detector (hilo inferencia) ─────────────────────────────
    def _on_result(self, result: dict, frame):
        """Actualiza el display en vivo; no toca la base de datos."""
        with self._result_lock:
            self._pending_result = result

    def _on_confirmed(self, result: dict, frame):
        """Llamado una única vez por detección confirmada; guarda en DB."""
        snap_path = self._save_snapshot(frame, result["label"])
        self.db.insert(result["fruit"], result["state"],
                       result["confidence"], snap_path)

    def _save_snapshot(self, frame, label: str) -> str:
        os.makedirs("data/snapshots", exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = f"data/snapshots/{label}_{ts}.jpg"
        cv2.imwrite(path, frame)
        return path

    # ── Loops de actualización (hilo principal Tkinter) ───────────────────────
    def _update_video_loop(self):
        frame = self.detector.get_frame()
        if frame is not None:
            self._render_frame(frame)
        self.after(33, self._update_video_loop)   # ~30 fps

    def _update_result_loop(self):
        with self._result_lock:
            result = self._pending_result
            self._pending_result = None

        if result:
            if result.get("label") == "indefinido":
                self._lbl_fruit.config(text="—", fg=TEXT_MAIN)
                self._lbl_state.config(text="—", fg=TEXT_DIM)
                self._lbl_conf.config(text="Confianza: —")
            else:
                fruit = result.get("fruit", "—")
                state = result.get("state", "—")
                conf  = result.get("confidence", 0)
                color = STATE_COLORS.get(state.lower(), TEXT_DIM)
                self._lbl_fruit.config(text=fruit or "—")
                self._lbl_state.config(text=state or "—", fg=color)
                self._lbl_conf.config(text=f"Confianza: {conf:.1%}")
            self._refresh_history()

        self.after(500, self._update_result_loop)

    def _render_frame(self, frame):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img     = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.delete("all")
        self._canvas.create_image(cw//2, ch//2, anchor="center", image=img)
        self._canvas._img = img   # evitar GC

    def _refresh_history(self):
        rows = self.db.get_recent(10)
        self._hist_text.config(state="normal")
        self._hist_text.delete("1.0", "end")
        for r in rows:
            ts, fruit, state, conf = r[1], r[2], r[3], r[4]
            line = f"{ts[:19]}  {fruit:<8} {state:<8} {conf:.0%}\n"
            self._hist_text.insert("end", line)
        self._hist_text.config(state="disabled")

    # ── Exportaciones ─────────────────────────────────────────────────────────
    def _export_pdf(self):
        rows    = self.db.get_all()
        summary = self.db.get_summary()
        path    = generate_pdf(rows, summary)
        messagebox.showinfo("PDF generado", f"Guardado en:\n{path}")

    def _export_excel(self):
        rows    = self.db.get_all()
        summary = self.db.get_summary()
        path    = generate_excel(rows, summary)
        messagebox.showinfo("Excel/CSV generado", f"Guardado en:\n{path}")

    def on_close(self):
        self.detector.stop()
        self.db.close()
        self.destroy()


def launch():
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
