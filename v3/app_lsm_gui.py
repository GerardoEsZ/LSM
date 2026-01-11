import os
# Silenciar logs de TensorFlow antes de importarlo
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# =========================
# CONFIGURACIÓN GENERAL
# =========================

MODEL_FILE = "modelo_gestos_bilstm.h5"
LABELS_FILE = "gestures_labels_mejorado.json"
DATA_FILE = "gestures_sequences_mejorado.npz"

SEQ_LEN = 60          # longitud de secuencia fija (frames)
FEATURE_DIM = 252     # 126 pos + 126 vel
MIN_SEQ_LEN = 10      # mínimo de frames para guardar secuencia

# Estabilidad de predicción
PROB_HISTORY_LEN = 5
PROB_THRESHOLD = 0.8
STABLE_REQUIRED = 6
COOLDOWN_FRAMES = 15

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 🎨 PALETA (tema morado/azul moderno)
COLOR_BG_MAIN = "#0f172a"      # azul muy oscuro
COLOR_CARD = "#111827"         # gris-azul
COLOR_CARD_SOFT = "#1f2937"
COLOR_ACCENT = "#8b5cf6"       # morado
COLOR_ACCENT_SOFT = "#a855f7"
COLOR_ACCENT_LIGHT = "#c4b5fd"
COLOR_TEXT = "#f9fafb"         # casi blanco
COLOR_SUBTEXT = "#9ca3af"      # gris
COLOR_SUCCESS = "#22c55e"
COLOR_WARN = "#eab308"
COLOR_ERROR = "#f97316"


# =========================
# FUNCIONES COMUNES
# =========================

def extraer_posiciones_dos_manos(results):
    """
    Devuelve vector (126,) con posiciones normalizadas de ambas manos:
    [21*3 mano derecha, 21*3 mano izquierda]
    Centrado en muñeca, normalizado por tamaño, izquierda espejada.
    """
    right_pts = np.zeros((21, 3), dtype=np.float32)
    left_pts = np.zeros((21, 3), dtype=np.float32)

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return np.concatenate([right_pts.flatten(), left_pts.flatten()])

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

        # centrar en muñeca
        wrist = pts[0].copy()
        pts -= wrist

        # normalizar por tamaño (distancia máxima en XY)
        dists = np.linalg.norm(pts[:, :2], axis=1)
        max_dist = np.max(dists)
        if max_dist > 0:
            pts /= max_dist

        if label.lower().startswith('r'):
            right_pts = pts
        else:
            # espejar mano izquierda en X
            pts[:, 0] = -pts[:, 0]
            left_pts = pts

    return np.concatenate([right_pts.flatten(), left_pts.flatten()])


def pad_sequence(seq, max_len, feature_dim):
    """Rellena o corta una secuencia (L, F) a (max_len, F) con ceros al final."""
    L = len(seq)
    out = np.zeros((max_len, feature_dim), dtype=np.float32)
    out[:min(L, max_len), :] = seq[:max_len]
    return out


def cargar_modelo_y_labels():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"No se encontró el modelo '{MODEL_FILE}'.\n\n"
            "Primero captura datos y entrena el modelo."
        )
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(
            f"No se encontró el archivo de etiquetas '{LABELS_FILE}'."
        )
    model = load_model(MODEL_FILE)
    with open(LABELS_FILE, "r") as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    return model, label_map


def obtener_info_modelo_dataset():
    lines = []
    if not os.path.exists(MODEL_FILE):
        lines.append(f"✖ Modelo NO encontrado: {MODEL_FILE}")
    else:
        lines.append(f"✔ Modelo encontrado: {MODEL_FILE}")

    label_map = {}
    if not os.path.exists(LABELS_FILE):
        lines.append(f"✖ Archivo de etiquetas NO encontrado: {LABELS_FILE}")
    else:
        with open(LABELS_FILE, "r") as f:
            label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}
        lines.append(f"\nClases ({len(label_map)}):")
        for i, name in label_map.items():
            lines.append(f"  [{i}] {name}")

    if not os.path.exists(DATA_FILE):
        lines.append(f"\n✖ Dataset NO encontrado: {DATA_FILE}")
    else:
        data = np.load(DATA_FILE)
        X = data["X"]
        y = data["y"]
        lines.append(f"\nSecuencias en dataset: {len(X)}")
        uniques, counts = np.unique(y, return_counts=True)
        lines.append("Distribución por clase:")
        for u, c in zip(uniques, counts):
            name = label_map.get(int(u), "desconocida")
            lines.append(f"  Label {u} ({name}): {c} secuencias")

    return "\n".join(lines)


def entrenar_modelo_desde_gui(parent):
    if not os.path.exists(DATA_FILE):
        messagebox.showerror(
            "Dataset inexistente",
            f"No se encontró el dataset '{DATA_FILE}'.\n\n"
            "Primero captura datos desde 'Capturar secuencias' en este menú."
        )
        return

    resp = messagebox.askyesno(
        "Entrenar modelo",
        "El entrenamiento puede tardar varios minutos.\n\n¿Deseas continuar?"
    )
    if not resp:
        return

    os.system("python entrenar_bilstm_mejorado.py")
    messagebox.showinfo(
        "Entrenamiento finalizado",
        "El entrenamiento ha terminado.\n"
        "Verifica la consola por si hubo errores.\n"
        "Si todo salió bien, el modelo se ha actualizado."
    )


# =========================
# VENTANA DE CAPTURA (TODO GUI)
# =========================

class CaptureApp(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Captura de secuencias - Dataset LSM")
        self.geometry("1220x720")
        self.configure(bg=COLOR_BG_MAIN)

        # Cargar dataset previo si existe
        self.X_list = []
        self.y_list = []
        if os.path.exists(DATA_FILE):
            data = np.load(DATA_FILE)
            self.X_list = list(data["X"])
            self.y_list = list(data["y"])

        # Cargar etiquetas previas si existen
        self.label_map = {}
        default_classes_str = ""
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, "r") as f:
                prev_labels = json.load(f)
            prev_labels = {int(k): v for k, v in prev_labels.items()}
            self.label_map = prev_labels
            default_classes_str = ",".join(prev_labels[i] for i in sorted(prev_labels.keys()))

        self.current_class_idx = 0
        self.recording = False
        self.seq_buffer = []
        self.prev_pos = np.zeros(126, dtype=np.float32)

        # MediaPipe y cámara
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Construir UI
        self._build_ui(default_classes_str)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Loop de video
        self.update_frame()

    def _build_ui(self, default_classes_str):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Root.TFrame", background=COLOR_BG_MAIN)
        style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        style.configure("Side.TFrame", background=COLOR_CARD)
        style.configure("Title.TLabel", background=COLOR_BG_MAIN,
                        foreground=COLOR_TEXT, font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", background=COLOR_BG_MAIN,
                        foreground=COLOR_SUBTEXT, font=("Segoe UI", 9))
        style.configure("SideTitle.TLabel", background=COLOR_CARD,
                        foreground=COLOR_TEXT, font=("Segoe UI", 13, "bold"))
        style.configure("Info.TLabel", background=COLOR_CARD,
                        foreground=COLOR_SUBTEXT, font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=COLOR_CARD,
                        foreground=COLOR_WARN, font=("Segoe UI", 10))
        style.configure("Big.TLabel", background=COLOR_CARD,
                        foreground=COLOR_TEXT, font=("Segoe UI", 18, "bold"))

        style.configure("Modern.TButton",
                        font=("Segoe UI", 10, "bold"),
                        foreground=COLOR_TEXT,
                        background=COLOR_ACCENT,
                        borderwidth=0,
                        padding=6)
        style.map("Modern.TButton",
                  background=[("active", COLOR_ACCENT_SOFT)],
                  foreground=[("disabled", "#6b7280")])

        # Barra superior
        top_bar = ttk.Frame(self, style="Root.TFrame")
        top_bar.pack(fill="x", padx=15, pady=(10, 5))

        ttk.Label(top_bar,
                  text="Captura de secuencias para el dataset LSM",
                  style="Title.TLabel").pack(side="left", padx=10, pady=5)

        ttk.Label(top_bar,
                  text="Define las clases y captura con la cámara",
                  style="Subtitle.TLabel").pack(side="left", padx=10)

        # Zona principal
        main_frame = ttk.Frame(self, style="Root.TFrame")
        main_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Video
        video_card = ttk.Frame(main_frame, style="Card.TFrame")
        video_card.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)

        self.video_label = ttk.Label(video_card, background=COLOR_CARD)
        self.video_label.pack(fill="both", expand=True, padx=12, pady=12)

        # Panel lateral
        side_card = ttk.Frame(main_frame, style="Side.TFrame")
        side_card.pack(side="right", fill="y", padx=(10, 0), pady=10)

        ttk.Label(side_card, text="Configuración de clases", style="SideTitle.TLabel").pack(
            anchor="w", padx=15, pady=(10, 5)
        )

        ttk.Label(side_card, text="Ingresa las clases separadas por comas (ej: A,B,C,Z,HOLA,SI):",
                  style="Info.TLabel").pack(anchor="w", padx=15)

        self.entry_clases = ttk.Entry(side_card)
        self.entry_clases.pack(fill="x", padx=15, pady=(2, 5))
        self.entry_clases.insert(0, default_classes_str)

        self.btn_aplicar_clases = ttk.Button(
            side_card, text="Aplicar clases / Guardar etiquetas",
            style="Modern.TButton", command=self.aplicar_clases
        )
        self.btn_aplicar_clases.pack(fill="x", padx=15, pady=(2, 10))

        self.label_lista_clases = ttk.Label(
            side_card, text="Clases actuales:\n(none)", style="Info.TLabel", justify="left"
        )
        self.label_lista_clases.pack(anchor="w", padx=15, pady=(0, 10))

        ttk.Label(side_card, text="Captura:", style="SideTitle.TLabel").pack(
            anchor="w", padx=15, pady=(10, 5)
        )

        self.label_clase_actual = ttk.Label(side_card, text="Clase actual: (sin definir)",
                                            style="Big.TLabel")
        self.label_clase_actual.pack(anchor="w", padx=15, pady=(0, 5))

        self.label_estado = ttk.Label(side_card,
                                      text="Estado: Esperando...",
                                      style="Status.TLabel")
        self.label_estado.pack(anchor="w", padx=15, pady=(0, 10))

        self.label_total = ttk.Label(side_card,
                                     text=f"Secuencias guardadas: {len(self.X_list)}",
                                     style="Info.TLabel")
        self.label_total.pack(anchor="w", padx=15, pady=(0, 10))

        btn_frame = ttk.Frame(side_card, style="Side.TFrame")
        btn_frame.pack(fill="x", padx=15, pady=(10, 5))

        self.btn_prev = ttk.Button(
            btn_frame, text="Clase anterior",
            style="Modern.TButton", command=self.clase_anterior
        )
        self.btn_prev.pack(fill="x", pady=3)

        self.btn_next = ttk.Button(
            btn_frame, text="Siguiente clase",
            style="Modern.TButton", command=self.clase_siguiente
        )
        self.btn_next.pack(fill="x", pady=3)

        self.btn_toggle_rec = ttk.Button(
            btn_frame, text="Iniciar grabación",
            style="Modern.TButton", command=self.toggle_recording
        )
        self.btn_toggle_rec.pack(fill="x", pady=(10, 3))

        self.btn_close = ttk.Button(
            btn_frame, text="Terminar y guardar",
            style="Modern.TButton", command=self.on_close
        )
        self.btn_close.pack(fill="x", pady=(10, 3))

        # Actualizar texto de clases
        self.actualizar_texto_clases()
        self.actualizar_clase_actual()

    # -------- manejo de clases --------

    def aplicar_clases(self):
        text = self.entry_clases.get().strip()
        if not text:
            messagebox.showerror("Error", "Debes ingresar al menos una clase.")
            return

        clases = [c.strip().upper() for c in text.split(",") if c.strip()]
        if not clases:
            messagebox.showerror("Error", "No se encontraron clases válidas.")
            return

        self.label_map = {i: nombre for i, nombre in enumerate(clases)}
        self.current_class_idx = 0

        # Guardar en JSON
        with open(LABELS_FILE, "w") as f:
            json.dump(self.label_map, f)

        self.actualizar_texto_clases()
        self.actualizar_clase_actual()
        messagebox.showinfo("Clases actualizadas",
                            f"Se guardaron {len(clases)} clases en {LABELS_FILE}.")

    def actualizar_texto_clases(self):
        if not self.label_map:
            self.label_lista_clases.config(text="Clases actuales:\n(none)")
            return

        lines = ["Clases actuales:"]
        for i, nombre in sorted(self.label_map.items()):
            lines.append(f"  [{i}] {nombre}")
        self.label_lista_clases.config(text="\n".join(lines))

    def actualizar_clase_actual(self):
        if not self.label_map:
            self.label_clase_actual.config(text="Clase actual: (sin definir)")
        else:
            nombre = self.label_map.get(self.current_class_idx, "(?)")
            self.label_clase_actual.config(text=f"Clase actual [{self.current_class_idx}]: {nombre}")

    def clase_siguiente(self):
        if not self.label_map:
            return
        self.current_class_idx = (self.current_class_idx + 1) % len(self.label_map)
        self.actualizar_clase_actual()

    def clase_anterior(self):
        if not self.label_map:
            return
        self.current_class_idx = (self.current_class_idx - 1) % len(self.label_map)
        self.actualizar_clase_actual()

    # -------- grabación --------

    def toggle_recording(self):
        if not self.label_map:
            messagebox.showerror(
                "Clases no definidas",
                "Primero define y aplica las clases antes de grabar."
            )
            return

        if not self.recording:
            # Empezar
            self.recording = True
            self.seq_buffer = []
            self.prev_pos = np.zeros(126, dtype=np.float32)
            self.btn_toggle_rec.config(text="Detener grabación")
            self.label_estado.config(
                text=f"Grabando secuencia para clase {self.current_class_idx}: {self.label_map[self.current_class_idx]}",
                foreground=COLOR_SUCCESS
            )
        else:
            # Detener y guardar
            self.recording = False
            self.btn_toggle_rec.config(text="Iniciar grabación")
            if len(self.seq_buffer) >= MIN_SEQ_LEN:
                seq_arr = np.array(self.seq_buffer, dtype=np.float32)
                seq_padded = pad_sequence(seq_arr, SEQ_LEN, seq_arr.shape[1])
                self.X_list.append(seq_padded)
                self.y_list.append(self.current_class_idx)
                self.label_estado.config(
                    text=f"Secuencia guardada ({len(self.seq_buffer)} frames)",
                    foreground=COLOR_SUCCESS
                )
                self.label_total.config(
                    text=f"Secuencias guardadas: {len(self.X_list)}"
                )
            else:
                self.label_estado.config(
                    text=f"Secuencia demasiado corta ({len(self.seq_buffer)} frames). No se guardó.",
                    foreground=COLOR_ERROR
                )
            self.seq_buffer = []

    # -------- loop de video --------

    def update_frame(self):
        if not hasattr(self, "cap") or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.label_estado.config(text="No se pudo leer frame de la cámara", foreground=COLOR_ERROR)
            self.after(30, self.update_frame)
            return

        frame_bgr = frame.copy()
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_bgr, hl, mp_hands.HAND_CONNECTIONS)

        # Extraer características
        curr_pos = extraer_posiciones_dos_manos(results)
        vel = curr_pos - self.prev_pos
        self.prev_pos = curr_pos.copy()
        features = np.concatenate([curr_pos, vel])

        if self.recording:
            self.seq_buffer.append(features)

        # Mostrar video en Tkinter
        display = cv2.flip(frame_bgr, 1)
        text_info = "Grabando" if self.recording else "Esperando..."
        cv2.putText(display, text_info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255) if self.recording else (200, 200, 200), 2)

        img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((820, 620))
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.after(30, self.update_frame)

    def on_close(self):
        # Guardar dataset si hay algo
        if self.X_list:
            X = np.array(self.X_list, dtype=np.float32)
            y = np.array(self.y_list, dtype=np.int32)
            np.savez(DATA_FILE, X=X, y=y)
            print(f"[Capture] Guardado dataset {DATA_FILE} con {len(X)} secuencias.")
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.mp_hands.close()
        except Exception:
            pass
        self.destroy()


# =========================
# VENTANA DE RECONOCIMIENTO
# =========================

class LSMApp(tk.Toplevel):
    def __init__(self, master, practice_mode_start=False):
        super().__init__(master)
        self.title("Reconocimiento de LSM - Demo Tesis")
        self.geometry("1220x720")
        self.configure(bg=COLOR_BG_MAIN)

        try:
            self.model, self.label_map = cargar_modelo_y_labels()
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            self.destroy()
            return

        self.letter_ids = [i for i, name in self.label_map.items() if len(name) == 1]
        self.word_ids = [i for i, name in self.label_map.items() if len(name) > 1]

        # Estado interno
        self.mode = "LETRAS"
        self.current_text = ""
        self.last_appended = None
        self.cooldown_counter = 0
        self.stable_label = None
        self.stable_count = 0
        self.prob_history = deque(maxlen=PROB_HISTORY_LEN)
        self.seq_buffer = deque(maxlen=SEQ_LEN)
        self.prev_pos = np.zeros(126, dtype=np.float32)

        # Práctica
        self.practice_mode = practice_mode_start
        self.practice_target = None

        # Cámara y manos
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Construir UI
        self._build_ui()

        if self.practice_mode:
            self.elegir_nueva_practica()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Animaciones pequeñas
        self._pulse_state = True
        self._glow_state = False
        self.animate_status_dot()
        self.animate_practice_label()

        # Loop de video
        self.update_frame()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Root.TFrame", background=COLOR_BG_MAIN)
        style.configure("Card.TFrame", background=COLOR_CARD, relief="flat")
        style.configure("Video.TFrame", background=COLOR_CARD)
        style.configure("Side.TFrame", background=COLOR_CARD)

        style.configure("Title.TLabel", background=COLOR_BG_MAIN,
                        foreground=COLOR_TEXT, font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", background=COLOR_BG_MAIN,
                        foreground=COLOR_SUBTEXT, font=("Segoe UI", 10))

        style.configure("SideTitle.TLabel", background=COLOR_CARD,
                        foreground=COLOR_TEXT, font=("Segoe UI", 13, "bold"))
        style.configure("Mode.TLabel", background=COLOR_CARD,
                        foreground=COLOR_ACCENT_LIGHT, font=("Segoe UI", 11, "bold"))
        style.configure("Big.TLabel", background=COLOR_CARD,
                        foreground=COLOR_TEXT, font=("Segoe UI", 22, "bold"))
        style.configure("Info.TLabel", background=COLOR_CARD,
                        foreground=COLOR_SUBTEXT, font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=COLOR_CARD,
                        foreground=COLOR_WARN, font=("Segoe UI", 10))
        style.configure("Result.TLabel", background=COLOR_CARD,
                        foreground=COLOR_WARN, font=("Segoe UI", 11, "bold"))

        style.configure("Modern.TButton",
                        font=("Segoe UI", 10, "bold"),
                        foreground=COLOR_TEXT,
                        background=COLOR_ACCENT,
                        borderwidth=0,
                        padding=6)
        style.map("Modern.TButton",
                  background=[("active", COLOR_ACCENT_SOFT)],
                  foreground=[("disabled", "#6b7280")])

        # Barra superior
        top_bar = ttk.Frame(self, style="Root.TFrame")
        top_bar.pack(fill="x", padx=15, pady=(10, 5))

        ttk.Label(top_bar,
                  text="Sistema de Reconocimiento de Lengua de Señas Mexicana",
                  style="Title.TLabel").pack(side="left", padx=10, pady=5)

        ttk.Label(top_bar,
                  text="Tiempo real · Modelo BiLSTM · Detección bimanual",
                  style="Subtitle.TLabel").pack(side="left", padx=10)

        # Indicador de cámara
        self.status_dot_canvas = tk.Canvas(top_bar, width=22, height=22,
                                           bg=COLOR_BG_MAIN, highlightthickness=0)
        self.status_dot_canvas.pack(side="right", padx=10)
        self.status_dot = self.status_dot_canvas.create_oval(4, 4, 18, 18,
                                                              fill=COLOR_SUCCESS, outline="")

        self.status_text = ttk.Label(top_bar,
                                     text="Cámara activa",
                                     style="Subtitle.TLabel")
        self.status_text.pack(side="right", padx=5)

        # Contenido principal
        main_frame = ttk.Frame(self, style="Root.TFrame")
        main_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Video
        video_card = ttk.Frame(main_frame, style="Video.TFrame")
        video_card.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)

        self.video_label = ttk.Label(video_card, background=COLOR_CARD)
        self.video_label.pack(fill="both", expand=True, padx=12, pady=12)

        # Panel lateral
        side_card = ttk.Frame(main_frame, style="Side.TFrame")
        side_card.pack(side="right", fill="y", padx=(10, 0), pady=10)

        ttk.Label(side_card, text="Panel de Control", style="SideTitle.TLabel").pack(
            anchor="w", padx=15, pady=(10, 5)
        )

        self.mode_label = ttk.Label(side_card, text=f"Modo actual: {self.mode}", style="Mode.TLabel")
        self.mode_label.pack(anchor="w", padx=15, pady=(0, 10))

        ttk.Label(side_card, text="Texto reconocido:", style="Info.TLabel").pack(
            anchor="w", padx=15
        )
        self.text_label = ttk.Label(side_card, text="", style="Big.TLabel", wraplength=320)
        self.text_label.pack(anchor="w", padx=15, pady=(0, 10))

        ttk.Label(side_card, text="Práctica:", style="Info.TLabel").pack(anchor="w", padx=15)
        self.practice_target_label = ttk.Label(side_card, text="Objetivo: (no activo)", style="Info.TLabel")
        self.practice_target_label.pack(anchor="w", padx=15)

        self.practice_result_label = ttk.Label(side_card, text="Resultado: -", style="Result.TLabel")
        self.practice_result_label.pack(anchor="w", padx=15, pady=(0, 10))

        self.status_label = ttk.Label(side_card, text="Estado: Iniciando...", style="Status.TLabel")
        self.status_label.pack(anchor="w", padx=15, pady=(5, 10))

        # Botones
        btn_frame = ttk.Frame(side_card, style="Side.TFrame")
        btn_frame.pack(fill="x", padx=15, pady=(10, 5))

        self.btn_mode = ttk.Button(btn_frame, text="Cambiar a PALABRAS",
                                   style="Modern.TButton", command=self.toggle_mode)
        self.btn_mode.pack(fill="x", pady=4)

        self.btn_clear = ttk.Button(btn_frame, text="Limpiar texto",
                                    style="Modern.TButton", command=self.clear_text)
        self.btn_clear.pack(fill="x", pady=4)

        self.btn_practice = ttk.Button(btn_frame, text="Activar / Desactivar práctica",
                                       style="Modern.TButton", command=self.toggle_practice)
        self.btn_practice.pack(fill="x", pady=4)

        self.btn_close = ttk.Button(btn_frame, text="Cerrar ventana",
                                    style="Modern.TButton", command=self.on_close)
        self.btn_close.pack(fill="x", pady=(12, 4))

    # ---------- Animaciones ----------

    def animate_status_dot(self):
        self._pulse_state = not self._pulse_state
        color = COLOR_SUCCESS if self._pulse_state else "#16a34a"
        self.status_dot_canvas.itemconfig(self.status_dot, fill=color)
        self.after(500, self.animate_status_dot)

    def animate_practice_label(self):
        if self.practice_mode:
            self._glow_state = not self._glow_state
            if "¡ACIERTO" in self.practice_result_label.cget("text"):
                fg = COLOR_ACCENT_LIGHT if self._glow_state else COLOR_SUCCESS
                self.practice_result_label.configure(foreground=fg)
        self.after(400, self.animate_practice_label)

    # ---------- Controles ----------

    def toggle_mode(self):
        self.mode = "PALABRAS" if self.mode == "LETRAS" else "LETRAS"
        self.mode_label.config(text=f"Modo actual: {self.mode}")
        self.btn_mode.config(text=f"Cambiar a {'LETRAS' if self.mode == 'PALABRAS' else 'PALABRAS'}")
        self.stable_label = None
        self.stable_count = 0
        self.last_appended = None
        self.prob_history.clear()
        self.status_label.config(text="Modo cambiado, estabilizando...", foreground=COLOR_WARN)

    def clear_text(self):
        self.current_text = ""
        self.text_label.config(text=self.current_text)
        self.last_appended = None
        self.stable_label = None
        self.stable_count = 0
        self.prob_history.clear()
        self.status_label.config(text="Texto limpiado", foreground=COLOR_WARN)

    def toggle_practice(self):
        self.practice_mode = not self.practice_mode
        if self.practice_mode:
            self.elegir_nueva_practica()
            self.status_label.config(text="Modo práctica ACTIVADO", foreground=COLOR_SUCCESS)
        else:
            self.practice_target = None
            self.practice_target_label.config(text="Objetivo: (no activo)")
            self.practice_result_label.config(text="Resultado: -", foreground=COLOR_WARN)
            self.status_label.config(text="Modo práctica DESACTIVADO", foreground=COLOR_WARN)

    def elegir_nueva_practica(self):
        ids = list(self.label_map.keys())
        self.practice_target = random.choice(ids)
        objetivo = self.label_map[self.practice_target]
        self.practice_target_label.config(text=f"Objetivo: {objetivo}")
        self.practice_result_label.config(text="Resultado: Esperando...", foreground=COLOR_WARN)

    def on_close(self):
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.mp_hands.close()
        except Exception:
            pass
        self.destroy()

    # ---------- Loop de video ----------

    def update_frame(self):
        if not hasattr(self, "cap") or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="No se pudo leer frame de la cámara", foreground=COLOR_ERROR)
            self.after(30, self.update_frame)
            return

        frame_bgr = frame.copy()
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_bgr, hl, mp_hands.HAND_CONNECTIONS)

        curr_pos = extraer_posiciones_dos_manos(results)
        vel = curr_pos - self.prev_pos
        self.prev_pos = curr_pos.copy()
        features = np.concatenate([curr_pos, vel])
        self.seq_buffer.append(features)

        pred_text = "Esperando..."
        if len(self.seq_buffer) == SEQ_LEN:
            X = np.array(self.seq_buffer, dtype=np.float32).reshape(1, SEQ_LEN, FEATURE_DIM)
            probs = self.model.predict(X, verbose=0)[0]
            self.prob_history.append(probs)
            probs_smooth = np.mean(np.array(self.prob_history), axis=0)

            mask = np.zeros_like(probs_smooth)
            idxs_validos = self.letter_ids if self.mode == "LETRAS" else self.word_ids

            if idxs_validos:
                for i in idxs_validos:
                    mask[i] = 1.0
                probs_smooth *= mask

                if probs_smooth.sum() > 0:
                    probs_smooth /= probs_smooth.sum()
                    pred_label = int(np.argmax(probs_smooth))
                    prob = float(np.max(probs_smooth))
                    nombre = self.label_map[pred_label]

                    if self.cooldown_counter == 0 and prob >= PROB_THRESHOLD:
                        if self.stable_label == pred_label:
                            self.stable_count += 1
                        else:
                            self.stable_label = pred_label
                            self.stable_count = 1

                        if self.stable_count >= STABLE_REQUIRED and pred_label != self.last_appended:
                            if not self.practice_mode:
                                if self.mode == "LETRAS":
                                    self.current_text += nombre
                                else:
                                    if self.current_text and not self.current_text.endswith(" "):
                                        self.current_text += " "
                                    self.current_text += nombre + " "
                                self.text_label.config(text=self.current_text)
                            else:
                                if self.practice_target is not None:
                                    if pred_label == self.practice_target:
                                        self.practice_result_label.config(
                                            text="Resultado: ¡ACIERTO!", foreground=COLOR_SUCCESS
                                        )
                                        self.after(1500, self.elegir_nueva_practica)
                                    else:
                                        self.practice_result_label.config(
                                            text="Resultado: NO COINCIDE", foreground=COLOR_ERROR
                                        )
                            self.last_appended = pred_label
                            self.cooldown_counter = COOLDOWN_FRAMES
                            self.stable_count = 0

                    pred_text = f"{nombre} ({prob:.2f})"
                else:
                    pred_text = "Sin clase válida en este modo"

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        display = cv2.flip(frame_bgr, 1)
        cv2.putText(display, pred_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((820, 620))
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.status_label.config(text=f"Predicción: {pred_text}", foreground=COLOR_WARN)

        self.after(30, self.update_frame)


# =========================
# VENTANA PRINCIPAL
# =========================

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LSM - Sistema de Reconocimiento (Tesis)")
        self.geometry("1000x650")
        self.configure(bg=COLOR_BG_MAIN)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Main.TFrame", background=COLOR_BG_MAIN)
        style.configure("MainTitle.TLabel", background=COLOR_BG_MAIN,
                        foreground=COLOR_TEXT, font=("Segoe UI", 18, "bold"))
        style.configure("MainSubtitle.TLabel", background=COLOR_BG_MAIN,
                        foreground=COLOR_SUBTEXT, font=("Segoe UI", 10))
        style.configure("MainButton.TButton",
                        font=("Segoe UI", 11, "bold"),
                        foreground=COLOR_TEXT,
                        background=COLOR_ACCENT,
                        borderwidth=0,
                        padding=8)
        style.map("MainButton.TButton",
                  background=[("active", COLOR_ACCENT_SOFT)],
                  foreground=[("disabled", "#6b7280")])

        main_frame = ttk.Frame(self, style="Main.TFrame", padding=20)
        main_frame.pack(fill="both", expand=True)

        ttk.Label(main_frame,
                  text="Sistema de Reconocimiento de Lengua de Señas Mexicana",
                  style="MainTitle.TLabel").pack(pady=(0, 5))

        ttk.Label(main_frame,
                  text="Proyecto de titulación · Ingeniería en Sistemas Computacionales",
                  style="MainSubtitle.TLabel").pack(pady=(0, 20))

        # Tarjeta de botones
        card = ttk.Frame(main_frame, style="Card.TFrame")
        card.pack(fill="x", padx=5, pady=(0, 15))

        btn_frame = ttk.Frame(card, style="Card.TFrame")
        btn_frame.pack(padx=20, pady=20, fill="x")

        ttk.Button(btn_frame, text="Capturar secuencias (dataset)",
                   style="MainButton.TButton",
                   command=self.abrir_captura
                   ).pack(fill="x", pady=5)

        ttk.Button(btn_frame, text="Entrenar modelo (BiLSTM mejorado)",
                   style="MainButton.TButton",
                   command=lambda: entrenar_modelo_desde_gui(self)
                   ).pack(fill="x", pady=5)

        ttk.Button(btn_frame, text="Reconocer señas (modo normal)",
                   style="MainButton.TButton",
                   command=self.abrir_reconocer
                   ).pack(fill="x", pady=5)

        ttk.Button(btn_frame, text="Modo práctica (juego de señas)",
                   style="MainButton.TButton",
                   command=self.abrir_practica
                   ).pack(fill="x", pady=5)

        ttk.Button(btn_frame, text="Salir",
                   style="MainButton.TButton",
                   command=self.on_close
                   ).pack(fill="x", pady=(10, 5))

        # Panel de info
        info_card = ttk.Frame(main_frame, style="Card.TFrame")
        info_card.pack(fill="both", expand=True, padx=5, pady=(5, 0))

        ttk.Label(info_card, text="Información del modelo y dataset",
                  style="SideTitle.TLabel").pack(anchor="w", padx=15, pady=(10, 5))

        self.info_text = tk.Text(info_card, bg=COLOR_CARD_SOFT, fg=COLOR_TEXT,
                                 font=("Consolas", 10), height=12, bd=0)
        self.info_text.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        ttk.Button(info_card, text="Actualizar información",
                   style="MainButton.TButton",
                   command=self.mostrar_info
                   ).pack(anchor="e", padx=15, pady=(0, 10))

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def abrir_captura(self):
        CaptureApp(self)

    def abrir_reconocer(self):
        LSMApp(self, practice_mode_start=False)

    def abrir_practica(self):
        LSMApp(self, practice_mode_start=True)

    def mostrar_info(self):
        info = obtener_info_modelo_dataset()
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, info)

    def on_close(self):
        self.destroy()


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
