import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical

NPZ_FILE = "gestures_sequences_mejorado.npz"
LABELS_FILE = "gestures_labels_mejorado.json"
MODEL_FILE = "modelo_gestos_bilstm.h5"

AUG_MULTIPLIER = 2   # cuántas muestras sintéticas por muestra real
NOISE_STD = 0.02     # desviación estándar del ruido

# ===================== Cargar datos =====================
data = np.load(NPZ_FILE)
X = data["X"]  # (N, T, F) con F=252
y = data["y"]

with open(LABELS_FILE, "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

num_classes = len(np.unique(y))
T = X.shape[1]
F = X.shape[2]

print(f"Secuencias reales: {len(X)}")
print(f"T = {T}, F = {F}, clases = {num_classes}")
print(f"Labels: {label_map}")

# ===================== Data augmentation =====================

X_aug = [X]
y_aug = [y]

for k in range(AUG_MULTIPLIER):
    ruido = np.random.normal(0, NOISE_STD, size=X.shape).astype(np.float32)
    escala = (1.0 + np.random.uniform(-0.05, 0.05, size=(X.shape[0], 1, 1))).astype(np.float32)
    X_noisy = (X * escala) + ruido

    # pequeño time-shift (rotar frames)
    shifts = np.random.randint(-3, 4, size=(X.shape[0],))
    X_shifted = np.empty_like(X_noisy)
    for i, s in enumerate(shifts):
        X_shifted[i] = np.roll(X_noisy[i], shift=s, axis=0)

    X_aug.append(X_shifted)
    y_aug.append(y)

X_all = np.concatenate(X_aug, axis=0)
y_all = np.concatenate(y_aug, axis=0)

print(f"Total tras augmentation: {len(X_all)} secuencias")

# One-hot
y_cat = to_categorical(y_all, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_cat, test_size=0.2, random_state=42
)

# ===================== Modelo BiLSTM =====================

model = Sequential([
    Masking(mask_value=0.0, input_shape=(T, F)),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nExactitud en test: {acc:.3f}")

model.save(MODEL_FILE)
print(f"Modelo guardado en {MODEL_FILE}")
