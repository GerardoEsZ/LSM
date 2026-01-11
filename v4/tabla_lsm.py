import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import numpy as np

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

LABELS_FILE = os.path.join(DATASET_DIR, "gestures_labels_mejorado.json")
NPZ_FILE = os.path.join(DATASET_DIR, "gestures_sequences_mejorado.npz")

# ================= CARGA DATOS =================
def cargar_datos():
    if not os.path.exists(LABELS_FILE):
        messagebox.showerror("Error", "No se encontró el archivo de etiquetas")
        return [], []

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    conteo = {}
    if os.path.exists(NPZ_FILE):
        data = np.load(NPZ_FILE, allow_pickle=True)
        y = data["y"]
        for label in y:
            conteo[label] = conteo.get(label, 0) + 1

    filas = []
    for k, v in labels.items():
        filas.append((k, v, conteo.get(int(k), 0)))

    return filas, labels

# ================= VENTANA =================
root = tk.Tk()
root.title("Señas registradas")
root.geometry("600x420")
root.resizable(False, False)

# ================= ESTILO =================
style = ttk.Style()
style.theme_use("default")

style.configure(
    "Treeview",
    font=("Segoe UI", 10),
    rowheight=26
)
style.configure(
    "Treeview.Heading",
    font=("Segoe UI", 11, "bold")
)

# ================= BUSCADOR =================
buscador = tk.Entry(root, font=("Segoe UI", 12))
buscador.pack(fill="x", padx=10, pady=8)
buscador.insert(0, "Buscar...")

# ================= TABLA =================
frame = tk.Frame(root)
frame.pack(fill="both", expand=True, padx=10)

columns = ("id", "nombre", "registros")
tabla = ttk.Treeview(
    frame,
    columns=columns,
    show="headings"
)

tabla.heading("id", text="ID")
tabla.heading("nombre", text="Letra / Palabra")
tabla.heading("registros", text="Registros")

tabla.column("id", width=60, anchor="center")
tabla.column("nombre", width=260)
tabla.column("registros", width=100, anchor="center")

scroll = ttk.Scrollbar(frame, orient="vertical", command=tabla.yview)
tabla.configure(yscrollcommand=scroll.set)

tabla.pack(side="left", fill="both", expand=True)
scroll.pack(side="right", fill="y")

# ================= CARGA =================
filas, labels = cargar_datos()

def cargar_tabla(filtro=""):
    tabla.delete(*tabla.get_children())
    for fila in filas:
        if filtro.lower() in fila[1].lower():
            tabla.insert("", "end", values=fila)

cargar_tabla()

# ================= FILTRO =================
def filtrar(*args):
    texto = buscador.get()
    if texto == "Buscar...":
        cargar_tabla()
    else:
        cargar_tabla(texto)

buscador.bind("<KeyRelease>", filtrar)

def limpiar_placeholder(e):
    if buscador.get() == "Buscar...":
        buscador.delete(0, tk.END)

buscador.bind("<FocusIn>", limpiar_placeholder)

# ================= BOTÓN CERRAR =================
tk.Button(
    root,
    text="Cerrar",
    font=("Segoe UI", 11, "bold"),
    command=root.destroy
).pack(pady=8)

root.mainloop()
