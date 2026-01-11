import numpy as np
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

NPZ_FILE = os.path.join(DATASET_DIR, "gestures_sequences_mejorado.npz")
LABELS_FILE = os.path.join(DATASET_DIR, "gestures_labels_mejorado.json")
MODEL_FILE = os.path.join(BASE_DIR, "modelo_lsm.h5")

SEQ_LEN = 60
FEATURE_DIM = 126   # 21 puntos * 3 * 2 manos

# ================= VALIDACIONES =================
if not os.path.exists(NPZ_FILE):
    raise FileNotFoundError("❌ No existe el dataset. Captura señas primero.")

if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError("❌ No existe gestures_labels_mejorado.json")

# ================= CARGA =================
data = np.load(NPZ_FILE, allow_pickle=True)
X = np.array(data["X"], dtype=np.float32)
y = np.array(data["y"], dtype=np.int32)

with open(LABELS_FILE, "r", encoding="utf-8") as f:
    label_map = json.load(f)

num_classes = len(label_map)

print("📦 Clases detectadas:", label_map)
print("📊 Total muestras:", len(X))

# ================= SANIDAD =================
if len(X) == 0:
    raise ValueError("❌ Dataset vacío")

if X.shape[1:] != (SEQ_LEN, FEATURE_DIM):
    raise ValueError(f"❌ Forma incorrecta del dataset: {X.shape}")

# ================= ONE-HOT =================
y_cat = to_categorical(y, num_classes=num_classes)

# ================= MODELO =================
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True),
                  input_shape=(SEQ_LEN, FEATURE_DIM)),
    Dropout(0.3),

    Bidirectional(LSTM(64)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= ENTRENAMIENTO =================
model.fit(
    X,
    y_cat,
    epochs=30,
    batch_size=8,
    shuffle=True
)

# ================= GUARDADO =================
model.save(MODEL_FILE)

print("\n✅ MODELO ENTRENADO Y GUARDADO EN:")
print(MODEL_FILE)
