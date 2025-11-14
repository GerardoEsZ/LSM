import cv2
import mediapipe as mp
import csv
import os

OUTPUT_FILE = "dataset_manos.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ==============================
# 1. CONFIGURACIÓN INICIAL
# ==============================

letras_input = input("Ingresa las letras de las señas separadas por comas (ej: A,B,C,D): ")
letras = [l.strip().upper() for l in letras_input.split(",") if l.strip()]

if not letras:
    print("No ingresaste letras válidas. Saliendo...")
    exit()

label_map = {letra: idx for idx, letra in enumerate(letras)}

print("\nMapa de clases:")
for letra, idx in label_map.items():
    print(f"  {letra} -> {idx}")

print("\nEn la ventana de la cámara:")
print("  - Coloca SOLO una mano (izquierda o derecha).")
print("  - Haz la seña de la letra que quieres capturar.")
print("  - Presiona la LETRA correspondiente para guardar una muestra.")
print("  - Presiona 'q' para salir.\n")

# ==============================
# 2. PREPARAR CSV
# ==============================

if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header += ["label", "letter", "hand"]  # mano: R o L
        writer.writerow(header)
        print(f"Creado archivo {OUTPUT_FILE} con encabezados.")
else:
    print(f"Se usarán datos existentes en {OUTPUT_FILE} (se agregarán filas).")

# ==============================
# 3. CÁMARA Y MEDIAPIPE
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,     # hasta 2 manos
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

        # Dibujar manos detectadas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        texto1 = "Presiona letra (" + ",".join(letras) + ") para GUARDAR"
        texto2 = "Presiona 'q' para salir"
        cv2.putText(frame, texto1, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, texto2, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Captura de señas - MediaPipe", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Saliendo...")
            break

        # Si presionaste alguna tecla
        if key != 255 and results.multi_hand_landmarks and results.multi_handedness:
            letra_presionada = chr(key).upper()

            if letra_presionada in label_map:
                # Usar SOLO la primera mano detectada
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness_info = results.multi_handedness[0].classification[0].label  # 'Left' o 'Right'
                hand_short = 'L' if handedness_info.lower().startswith('l') else 'R'

                fila = []
                for lm in hand_landmarks.landmark:
                    fila += [lm.x, lm.y, lm.z]

                label_numerica = label_map[letra_presionada]
                fila.append(label_numerica)
                fila.append(letra_presionada)
                fila.append(hand_short)

                with open(OUTPUT_FILE, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(fila)

                print(f"Muestra guardada: letra={letra_presionada}, clase={label_numerica}, mano={hand_short}")
            else:
                print("Tecla presionada no está en la lista de letras definidas.")

cap.release()
cv2.destroyAllWindows()
