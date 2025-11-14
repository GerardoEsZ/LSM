import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import json
from collections import deque

MODEL_FILE = "modelo_senas_manos.h5"
LABELS_FILE = "labels.json"

model = load_model(MODEL_FILE)

with open(LABELS_FILE, "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

print("Mapa de etiquetas cargado:", label_map)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalizar_con_mano_row(row, hand):
    pts = np.array(row).reshape(21, 3)
    wrist = pts[0].copy()
    pts = pts - wrist

    if str(hand).upper() == 'L':
        pts[:, 0] = -pts[:, 0]

    dists = np.linalg.norm(pts[:, :2], axis=1)
    max_dist = np.max(dists)
    if max_dist > 0:
        pts = pts / max_dist

    return pts.flatten()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# Historial para suavizar
history_size = 7
pred_history = deque(maxlen=history_size)

current_text = ""          # texto que vamos construyendo
last_appended_label = None
stable_label = None
stable_frames = 0
required_stable_frames = 8  # cuántos frames debe mantenerse una letra antes de añadirla

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame de la cámara.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        texto_prediccion = "No se detecta mano"

        key = cv2.waitKey(1) & 0xFF

        # Controles de texto
        if key == ord(' '):  # espacio
            current_text += " "
        if key == ord('c'):  # clear
            current_text = ""
        if key == ord('q'):
            break

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_info = results.multi_handedness[0].classification[0].label  # 'Left' o 'Right'
            hand_short = 'L' if handedness_info.lower().startswith('l') else 'R'

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            row = []
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]

            X = np.array(normalizar_con_mano_row(row, hand_short)).reshape(1, -1)

            preds = model.predict(X, verbose=0)
            pred_label = int(np.argmax(preds))
            prob = float(np.max(preds))
            letter = label_map.get(pred_label, "?")

            pred_history.append(pred_label)
            counts = {lbl: pred_history.count(lbl) for lbl in set(pred_history)}
            smooth_label = max(counts, key=counts.get)
            smooth_letter = label_map.get(smooth_label, "?")

            # Seguimiento de estabilidad
            if stable_label == smooth_label:
                stable_frames += 1
            else:
                stable_label = smooth_label
                stable_frames = 1

            # Si la letra es estable varios frames y es suficientemente segura
            if stable_frames >= required_stable_frames and prob > 0.5:
                if last_appended_label != smooth_label:
                    current_text += smooth_letter
                    last_appended_label = smooth_label

            texto_prediccion = f"Letra: {smooth_letter} ({prob:.2f}) Mano: {hand_short}"
        else:
            pred_history.clear()
            stable_label = None
            stable_frames = 0

        cv2.putText(frame, texto_prediccion, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Texto: {current_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "SPACE=espacio, C=limpiar, Q=salir", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        cv2.imshow("Reconocimiento de señas (letras y palabras)", frame)

cap.release()
cv2.destroyAllWindows()
