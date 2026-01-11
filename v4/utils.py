import json
import os
import numpy as np

SEQ_LEN = 60
FEATURES = 252  # 2 manos x 21 puntos x (x,y,z)

def pad_or_sample(seq):
    seq = np.array(seq)
    length = len(seq)

    if length > SEQ_LEN:
        idx = np.linspace(0, length - 1, SEQ_LEN).astype(int)
        return seq[idx]
    elif length < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - length, FEATURES))
        return np.vstack([seq, pad])
    return seq

def load_classes():
    if not os.path.exists("dataset/clases.json"):
        return {}
    with open("dataset/clases.json", "r") as f:
        return json.load(f)

def save_classes(data):
    os.makedirs("dataset", exist_ok=True)
    with open("dataset/clases.json", "w") as f:
        json.dump(data, f, indent=4)
