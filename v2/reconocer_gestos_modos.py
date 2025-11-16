import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import json
from collections import deque

MODEL_FILE = "modelo_gestos_lstm.h5"
LABELS_FILE = "gestures_labels.json"
SEQ_LEN = 60  # igual que MAX_SEQ_LEN en captura

# Parámetros de estabilidad
STABLE_REQUIRED = 8     # frames estables para aceptar
PROB_THRESHOLD = 0.8    # probabilidad mínima
COOLDOWN_FRAMES = 15    # frames de "descanso" después de escribir

model = load_model(MODEL_FILE)

with open(LABELS_FILE, "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

print("Clases:", label_map)

# IDs de letras (1 carácter) y palabras (>1)
letter_ids = [i for i, name in label_map.items() if len(name) == 1]
word_ids = [i for i, name in label_map.items() if len(name) > 1]
print("IDs letras:", letter_ids)
print("IDs palabras:", word_ids)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extraer_caracteristicas_dos_manos(results):
    right_pts = np.zeros((21, 3), dtype=np.float32)
    left_pts = np.zeros((21, 3), dtype=np.float32)

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return np.concatenate([right_pts.flatten(), left_pts.flatten()])

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

        wrist = pts[0].copy()
        pts -= wrist

        dists = np.linalg.norm(pts[:, :2], axis=1)
        max_dist = np.max(dists)
        if max_dist > 0:
            pts /= max_dist

        if label.lower().startswith('r'):
            right_pts = pts
        else:
            pts[:, 0] = -pts[:, 0]
            left_pts = pts

    return np.concatenate([right_pts.flatten(), left_pts.flatten()])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

seq_buffer = deque(maxlen=SEQ_LEN)
mode = "LETRAS"   # o "PALABRAS"
current_text = ""
last_appended = None
stable_label = None
stable_count = 0
cooldown_counter = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame.")
            break

        frame_bgr = frame.copy()
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Dibujar manos
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_bgr, hl, mp_hands.HAND_CONNECTIONS)

        feat_vec = extraer_caracteristicas_dos_manos(results)
        seq_buffer.append(feat_vec)

        display = cv2.flip(frame_bgr, 1)

        # Barra superior
        cv2.rectangle(display, (0, 0), (display.shape[1], 130), (0, 0, 0), -1)
        cv2.putText(display, f"Modo: {mode} (m=cambiar)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        pred_text = "Esperando secuencia..."

        # Predicción solo si tenemos SEQ_LEN frames y no estamos en cooldown
        if len(seq_buffer) == SEQ_LEN and cooldown_counter == 0:
            X = np.array(seq_buffer, dtype=np.float32).reshape(1, SEQ_LEN, -1)
            preds = model.predict(X, verbose=0)[0]

            # Filtrar por modo
            mask = np.zeros_like(preds)
            idxs_validos = letter_ids if mode == "LETRAS" else word_ids

            if idxs_validos:
                for i in idxs_validos:
                    mask[i] = 1.0
                preds *= mask

                if preds.sum() > 0:
                    preds /= preds.sum()
                    pred_label = int(np.argmax(preds))
                    prob = float(np.max(preds))
                    nombre = label_map[pred_label]

                    # Si la prob es baja, reseteamos estabilidad
                    if prob < PROB_THRESHOLD:
                        stable_label = None
                        stable_count = 0
                    else:
                        # Seguimos acumulando estabilidad
                        if stable_label == pred_label:
                            stable_count += 1
                        else:
                            stable_label = pred_label
                            stable_count = 1

                        # Si llegó a frames suficientes y no es la última añadida
                        if stable_count >= STABLE_REQUIRED and pred_label != last_appended:
                            if mode == "LETRAS":
                                current_text += nombre
                            else:
                                if current_text and not current_text.endswith(" "):
                                    current_text += " "
                                current_text += nombre + " "
                            last_appended = pred_label
                            cooldown_counter = COOLDOWN_FRAMES
                            stable_count = 0

                    pred_text = f"{nombre} ({prob:.2f})"
                else:
                    pred_text = "Sin clase válida en este modo"
            else:
                pred_text = "No hay clases para este modo"

        # Reducir cooldown poco a poco
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # Textos en pantalla
        cv2.putText(display, f"Pred: {pred_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Texto: {current_text}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "SPACE=espacio, c=limpiar, m=modo, q=salir", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        # Mostrar la letra/palabra grande en el centro si hay predicción
        if "(" in pred_text:
            main_label = pred_text.split("(")[0].replace("Pred:", "").strip()
            h, w, _ = display.shape
            cv2.putText(display, main_label, (int(w/2) - 80, int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5)

        cv2.imshow("Reconocimiento de gestos (2 manos, movimiento)", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('m'):
            mode = "PALABRAS" if mode == "LETRAS" else "LETRAS"
            stable_label = None
            stable_count = 0
            last_appended = None
        if key == ord(' '):
            current_text += " "
        if key == ord('c'):
            current_text = ""
            last_appended = None
            stable_label = None
            stable_count = 0
            cooldown_counter = 0

cap.release()
cv2.destroyAllWindows()
