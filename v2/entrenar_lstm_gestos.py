import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical

NPZ_FILE = "gestures_sequences.npz"
LABELS_FILE = "gestures_labels.json"
MODEL_FILE = "modelo_gestos_lstm.h5"

# Cargar datos
data = np.load(NPZ_FILE)
X = data["X"]  # (N, T, F)
y = data["y"]  # (N,)

with open(LABELS_FILE, "r") as f:
    label_map = json.load(f)

num_classes = len(np.unique(y))
T = X.shape[1]
F = X.shape[2]

print(f"Secuencias: {len(X)}")
print(f"Longitud de secuencia (T): {T}")
print(f"Número de features (F): {F}")
print(f"Clases: {num_classes}, labels: {label_map}")

y_cat = to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# Modelo LSTM
model = Sequential([
    Masking(mask_value=0.0, input_shape=(T, F)),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
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
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nExactitud en test: {acc:.2f}")

model.save(MODEL_FILE)
print(f"Modelo guardado en {MODEL_FILE}")
