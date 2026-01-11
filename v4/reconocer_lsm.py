import cv2
import mediapipe as mp
import numpy as np
import json
import os
from collections import deque
from tensorflow.keras.models import load_model

# ================= CONFIG =================
SEQ_LEN = 60
FEATURE_DIM = 126

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

MODEL_FILE = os.path.join(BASE_DIR, "modelo_lsm.h5")
LABELS_FILE = os.path.join(DATASET_DIR, "gestures_labels_mejorado.json")

# ================= VALIDACIONES =================
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("❌ No se encontró el modelo entrenado")

if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError("❌ No se encontró gestures_labels_mejorado.json")

# ================= CARGA =================
model = load_model(MODEL_FILE)

with open(LABELS_FILE, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# Convertir keys a int
label_map = {int(k): v for k, v in label_map.items()}
print("📌 Clases:", label_map)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ================= FUNCIONES =================
def extraer_dos_manos(results):
    right = np.zeros((21, 3), dtype=np.float32)
    left = np.zeros((21, 3), dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, hand in zip(results.multi_hand_landmarks,
                            results.multi_handedness):
            puntos = np.array([[p.x, p.y, p.z] for p in lm.landmark],
                              dtype=np.float32)

            if hand.classification[0].label == "Right":
                right = puntos
            else:
                left = puntos

    return np.concatenate([right.flatten(), left.flatten()])

# ================= CAMARA =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ No se pudo abrir la cámara")

buffer = deque(maxlen=SEQ_LEN)
texto_actual = ""

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for h in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, h,
                                       mp_hands.HAND_CONNECTIONS)

        features = extraer_dos_manos(results)
        buffer.append(features)

        # ================= PREDICCIÓN =================
        if len(buffer) == SEQ_LEN:
            X = np.array(buffer, dtype=np.float32)
            X = X.reshape(1, SEQ_LEN, FEATURE_DIM)

            probs = model.predict(X, verbose=0)[0]
            idx = int(np.argmax(probs))
            conf = float(np.max(probs))

            if idx in label_map and conf > 0.75:
                palabra = label_map[idx]
                texto_actual = palabra
            else:
                texto_actual = "..."

        # ================= UI =================
        cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)

        cv2.putText(frame, "Reconocimiento LSM",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)

        cv2.putText(frame, f"Detectado: {texto_actual}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        cv2.putText(frame, "Q = Salir",
                    (500, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

        cv2.imshow("Reconocer LSM", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
