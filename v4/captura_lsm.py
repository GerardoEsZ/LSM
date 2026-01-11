import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
import tkinter as tk
from tkinter import simpledialog, messagebox

# ================= CONFIG =================
SEQ_LEN = 60
POINTS_PER_HAND = 21
FEATURES_PER_HAND = POINTS_PER_HAND * 3
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # dos manos

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

NPZ_FILE = os.path.join(DATASET_DIR, "gestures_sequences_mejorado.npz")
LABELS_FILE = os.path.join(DATASET_DIR, "gestures_labels_mejorado.json")

# ================= CARGA SEGURA =================
X, y = [], []

if os.path.exists(NPZ_FILE):
    try:
        data = np.load(NPZ_FILE, allow_pickle=True)
        X = list(data["X"])
        y = list(data["y"])
    except:
        X, y = [], []

labels = {}
if os.path.exists(LABELS_FILE):
    try:
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                labels = json.loads(content)
    except:
        labels = {}

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ================= FUNCIONES =================
def obtener_id(nombre):
    global labels

    for k, v in labels.items():
        if v.lower() == nombre.lower():
            return int(k)

    nuevo_id = max([int(k) for k in labels.keys()], default=-1) + 1
    labels[str(nuevo_id)] = nombre
    return nuevo_id


def extraer_dos_manos(results):
    right = np.zeros((POINTS_PER_HAND, 3), dtype=np.float32)
    left = np.zeros((POINTS_PER_HAND, 3), dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
            puntos = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            if hand.classification[0].label == "Right":
                right = puntos
            else:
                left = puntos

    return np.concatenate([right.flatten(), left.flatten()])


def capturar_secuencia(nombre):
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=2)

    # ===== CUENTA REGRESIVA =====
    inicio = time.time()
    while time.time() - inicio < 5:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        restante = 5 - int(time.time() - inicio)

        cv2.putText(frame, f"Prepárate: {restante}",
                    (180, 260), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 4)

        cv2.imshow("Captura LSM", frame)
        cv2.waitKey(1)

    # ===== CAPTURA =====
    secuencia = []
    frame_count = 0

    while frame_count < SEQ_LEN:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for h in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        features = extraer_dos_manos(results)
        secuencia.append(features)
        frame_count += 1

        # ===== BARRA DE PROGRESO =====
        progreso = int((frame_count / SEQ_LEN) * 400)
        cv2.rectangle(frame, (50, 430), (50 + progreso, 460), (0, 255, 0), -1)
        cv2.rectangle(frame, (50, 430), (450, 460), (255, 255, 255), 2)

        cv2.putText(frame, f"Grabando: {nombre}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        cv2.putText(frame, f"Frames: {frame_count}/{SEQ_LEN}",
                    (50, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        cv2.imshow("Captura LSM", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    return np.array(secuencia, dtype=np.float32)


# ================= INTERFAZ =================
root = tk.Tk()
root.withdraw()

nombre = simpledialog.askstring("Registrar seña", "Ingresa letra o palabra:")
if not nombre:
    messagebox.showerror("Error", "No se ingresó ninguna seña")
    exit()

label_id = obtener_id(nombre)
secuencia = capturar_secuencia(nombre)

X.append(secuencia)
y.append(label_id)

np.savez(NPZ_FILE, X=np.array(X, dtype=object), y=np.array(y))

with open(LABELS_FILE, "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=4, ensure_ascii=False)

messagebox.showinfo("Éxito", f"Seña '{nombre}' guardada correctamente")
