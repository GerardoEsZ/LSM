import cv2
import mediapipe as mp
import numpy as np
import json
import os

# ==============================
# CONFIGURACIÓN
# ==============================

MAX_SEQ_LEN = 60       # máximo de frames por secuencia (~2s)
MIN_SEQ_LEN = 10       # mínimo de frames para guardar

OUTPUT_NPZ = "gestures_sequences_mejorado.npz"
OUTPUT_LABELS = "gestures_labels_mejorado.json"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ==============================
# 1. PEDIR CLASES
# ==============================

clases_input = input("Ingresa las clases (letras/palabras) separadas por comas (ej: A,B,C,Z,HOLA,SI,NO): ")
clases = [c.strip().upper() for c in clases_input.split(",") if c.strip()]

if not clases:
    print("No ingresaste clases válidas. Saliendo...")
    exit()

label_map = {i: nombre for i, nombre in enumerate(clases)}

print("\nClases disponibles (indice : nombre):")
for i, nombre in label_map.items():
    print(f"  {i}: {nombre}")

# Cargar secuencias anteriores si existen
if os.path.exists(OUTPUT_NPZ):
    data = np.load(OUTPUT_NPZ)
    X_list = list(data["X"])
    y_list = list(data["y"])
    print(f"\nSe encontraron {len(X_list)} secuencias previas. Se agregarán más.")
else:
    X_list = []
    y_list = []
    print("\nNo hay secuencias previas. Se creará un archivo nuevo.")

# Guardar mapa de etiquetas
with open(OUTPUT_LABELS, "w") as f:
    json.dump(label_map, f)
print(f"\nMapa de etiquetas guardado en {OUTPUT_LABELS}: {label_map}\n")

# ==============================
# 2. FUNCIONES AUXILIARES
# ==============================

def extraer_posiciones_dos_manos(results):
    """
    Devuelve vector (126,) con posiciones normalizadas de ambas manos:
    [21*3 mano derecha, 21*3 mano izquierda], centradas en la muñeca y
    normalizadas por tamaño. Mano izquierda espejada en X.
    """
    right_pts = np.zeros((21, 3), dtype=np.float32)
    left_pts = np.zeros((21, 3), dtype=np.float32)

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return np.concatenate([right_pts.flatten(), left_pts.flatten()])

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label  # 'Left' o 'Right'
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

        # Centrar en muñeca
        wrist = pts[0].copy()
        pts -= wrist

        # Normalizar por tamaño
        dists = np.linalg.norm(pts[:, :2], axis=1)
        max_dist = np.max(dists)
        if max_dist > 0:
            pts /= max_dist

        if label.lower().startswith('r'):
            right_pts = pts
        else:
            pts[:, 0] = -pts[:, 0]  # espejar mano izquierda
            left_pts = pts

    return np.concatenate([right_pts.flatten(), left_pts.flatten()])

def pad_sequence(seq, max_len, feature_dim):
    L = len(seq)
    out = np.zeros((max_len, feature_dim), dtype=np.float32)
    out[:min(L, max_len), :] = seq[:max_len]
    return out

# ==============================
# 3. CAPTURA
# ==============================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("\nControles:")
print("  N = siguiente clase")
print("  P = clase anterior")
print("  G = empezar/detener grabación de la clase actual")
print(f"  MAX_SEQ_LEN = {MAX_SEQ_LEN}, MIN_SEQ_LEN = {MIN_SEQ_LEN}")
print("  Q = salir\n")

current_class_idx = 0  # clase seleccionada

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    recording = False
    seq_buffer = []

    # Usamos prev_pos para calcular velocidad solo cuando grabamos
    prev_pos = np.zeros(126, dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame de la cámara.")
            break

        frame_bgr = frame.copy()
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Dibujar manos
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_bgr, hl, mp_hands.HAND_CONNECTIONS)

        display = cv2.flip(frame_bgr, 1)

        # Barra superior
        cv2.rectangle(display, (0, 0), (display.shape[1], 100), (0, 0, 0), -1)
        cv2.putText(display, f"Clase actual [{current_class_idx}]: {label_map[current_class_idx]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, "N/P=clase  G=grabar/stop  Q=salir",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        # Siempre calculamos posición actual
        curr_pos = extraer_posiciones_dos_manos(results)

        if recording:
            # Al grabar, calculamos velocidad y guardamos features
            vel = curr_pos - prev_pos
            features = np.concatenate([curr_pos, vel])  # 126 pos + 126 vel = 252
            seq_buffer.append(features)

            cv2.putText(display, f"Grabando frames: {len(seq_buffer)}/{MAX_SEQ_LEN}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Actualizamos prev_pos para el siguiente frame
            prev_pos = curr_pos.copy()

            # Si llegamos al máximo, detenemos y guardamos
            if len(seq_buffer) >= MAX_SEQ_LEN:
                recording = False
                if len(seq_buffer) >= MIN_SEQ_LEN:
                    seq_arr = np.array(seq_buffer, dtype=np.float32)
                    seq_padded = pad_sequence(seq_arr, MAX_SEQ_LEN, seq_arr.shape[1])
                    X_list.append(seq_padded)
                    y_list.append(current_class_idx)
                    print(f"✅ Secuencia guardada (clase {current_class_idx}: {label_map[current_class_idx]}), total={len(X_list)}")
                else:
                    print("⚠ Secuencia demasiado corta. No se guardó.")
                seq_buffer = []

        cv2.imshow("Captura secuencias mejorada (manual)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue

        char = chr(key).lower()

        if char == 'q':
            print("Saliendo...")
            break

        # Cambiar clase
        if char == 'n':
            current_class_idx = (current_class_idx + 1) % len(label_map)
        elif char == 'p':
            current_class_idx = (current_class_idx - 1) % len(label_map)

        # Empezar / detener grabación manual con G
        elif char == 'g':
            if not recording:
                recording = True
                seq_buffer = []
                # Al iniciar grabación, sincronizamos prev_pos con la posición actual,
                # para que el primer frame tenga velocidad ~0
                prev_pos = curr_pos.copy()
                print(f"🎬 Iniciando grabación para clase [{current_class_idx}] {label_map[current_class_idx]}")
            else:
                recording = False
                if len(seq_buffer) >= MIN_SEQ_LEN:
                    seq_arr = np.array(seq_buffer, dtype=np.float32)
                    seq_padded = pad_sequence(seq_arr, MAX_SEQ_LEN, seq_arr.shape[1])
                    X_list.append(seq_padded)
                    y_list.append(current_class_idx)
                    print(f"✅ Secuencia guardada (clase {current_class_idx}: {label_map[current_class_idx]}), total={len(X_list)}")
                else:
                    print("⚠ Secuencia demasiado corta. No se guardó.")
                seq_buffer = []

cap.release()
cv2.destroyAllWindows()

# Guardar dataset
if X_list:
    X = np.array(X_list, dtype=np.float32)  # (N, MAX_SEQ_LEN, 252)
    y = np.array(y_list, dtype=np.int32)
    np.savez(OUTPUT_NPZ, X=X, y=y)
    print(f"\nGuardado archivo {OUTPUT_NPZ} con {len(X)} secuencias.")
else:
    print("No se guardó ninguna secuencia.")
