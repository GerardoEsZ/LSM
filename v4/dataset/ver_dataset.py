import numpy as np
import json

X = np.load("dataset/data.npy")
y = np.load("dataset/labels.npy")

with open("dataset/classes.json", "r", encoding="utf-8") as f:
    classes = json.load(f)

print("📦 Total de señas guardadas:", len(X))
print("📐 Forma de X:", X.shape)
print("🏷️ Etiquetas:", y[:10])
print("🧠 Clases registradas:", classes)
