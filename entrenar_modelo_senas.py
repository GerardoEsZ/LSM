import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import json

DATA_FILE = "dataset_manos.csv"

# ==============================
# 1. CARGAR DATASET
# ==============================

df = pd.read_csv(DATA_FILE)

coord_cols = [c for c in df.columns if c not in ["label", "letter", "hand"]]

X_raw = df[coord_cols].values
y = df["label"].values.astype(int)
hand_col = df["hand"].values  # 'R' o 'L'

num_classes = len(np.unique(y))
print(f"Total muestras: {len(X_raw)}")
print(f"Clases encontradas: {num_classes} -> {np.unique(y)}")

print("\nMuestras por clase:")
print(df["label"].value_counts().sort_index())

# ==============================
# 2. NORMALIZACIÓN CON MANO
# ==============================

def normalizar_con_mano(X, hands):
    """
    X: (N, 63)  -> 21 puntos * 3 coords
    hands: array de 'R' o 'L' para cada fila
    Centra en muñeca y espeja X si es mano izquierda.
    """
    X_norm = []
    for row, hand in zip(X, hands):
        pts = row.reshape(21, 3)
        wrist = pts[0].copy()
        pts = pts - wrist

        # Si es mano izquierda, espejamos X
        if str(hand).upper() == 'L':
            pts[:, 0] = -pts[:, 0]

        dists = np.linalg.norm(pts[:, :2], axis=1)
        max_dist = np.max(dists)
        if max_dist > 0:
            pts = pts / max_dist

        X_norm.append(pts.flatten())
    return np.array(X_norm)

X = normalizar_con_mano(X_raw, hand_col)

# ==============================
# 3. TRAIN / TEST
# ==============================

test_size = 0.2
if len(X) * test_size < num_classes:
    test_size = max(num_classes / len(X) + 0.05, 0.3)
    print(f"\nAjustando test_size a {test_size:.2f} para evitar errores.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

input_dim = X_train.shape[1]
print(f"\nDimensión de entrada: {input_dim}")
print(f"Muestras train: {len(X_train)}, test: {len(X_test)}")

# ==============================
# 4. MODELO
# ==============================

model = Sequential([
    Dense(128, activation="relu", input_shape=(input_dim,)),
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

# ==============================
# 5. ENTRENAR
# ==============================

history = model.fit(
    X_train,
    y_train_cat,
    epochs=60,
    batch_size=16,
    validation_data=(X_test, y_test_cat),
    verbose=1
)

# ==============================
# 6. EVALUAR Y GUARDAR
# ==============================

loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nExactitud en test: {acc:.2f}")

MODEL_FILE = "modelo_senas_manos.h5"
model.save(MODEL_FILE)
print(f"Modelo guardado en {MODEL_FILE}")

labels_df = df[["label", "letter"]].drop_duplicates().sort_values("label")
label_map = {int(row["label"]): row["letter"] for _, row in labels_df.iterrows()}

with open("labels.json", "w") as f:
    json.dump(label_map, f)

print("\nMapa de etiquetas guardado en labels.json:")
print(label_map)
