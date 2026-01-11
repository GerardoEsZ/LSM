import tkinter as tk
from tkinter import messagebox
import subprocess
import os

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAPTURA = os.path.join(BASE_DIR, "captura_lsm.py")
RECONOCER = os.path.join(BASE_DIR, "reconocer_lsm.py")
ENTRENAR = os.path.join(BASE_DIR, "entrenar_lsm.py")
TABLA = os.path.join(BASE_DIR, "tabla_lsm.py")  # nuevo archivo

# ================= VENTANA =================
root = tk.Tk()
root.title("Sistema de Lenguaje de Señas Mexicano")
root.geometry("900x550")
root.resizable(False, False)

# ================= TEMAS =================
light_theme = {
    "bg": "#0d47a1",
    "panel": "#1565c0",
    "btn": "#1976d2",
    "hover": "#1e88e5",
    "text": "white"
}

dark_theme = {
    "bg": "#121212",
    "panel": "#1e1e1e",
    "btn": "#2a2a2a",
    "hover": "#333333",
    "text": "white"
}

current_theme = light_theme

# ================= FUNCIONES =================
def aplicar_tema():
    root.configure(bg=current_theme["bg"])
    panel.configure(bg=current_theme["panel"])
    title.configure(bg=current_theme["panel"], fg=current_theme["text"])
    for b in botones:
        b.configure(bg=current_theme["btn"], fg=current_theme["text"])
    if 'footer' in globals():
        footer.configure(bg=current_theme["bg"], fg="white")

def toggle_theme():
    global current_theme
    current_theme = dark_theme if current_theme == light_theme else light_theme
    aplicar_tema()

def hover_on(e):
    e.widget.config(bg=current_theme["hover"])

def hover_off(e):
    e.widget.config(bg=current_theme["btn"])

def ejecutar(script):
    if not os.path.exists(script):
        messagebox.showerror("Error", f"No se encontró:\n{script}")
        return
    subprocess.Popen(["python", script])

# ================= PANEL =================
panel = tk.Frame(root, bg=current_theme["panel"])
panel.place(relx=0.5, rely=0.5, anchor="center", width=600, height=420)

title = tk.Label(
    panel,
    text="Lenguaje de Señas Mexicano",
    font=("Segoe UI", 22, "bold"),
    bg=current_theme["panel"],
    fg=current_theme["text"]
)
title.pack(pady=20)

# ================= BOTONES =================
botones = []

def crear_boton(texto, comando):
    btn = tk.Button(
        panel,
        text=texto,
        font=("Segoe UI", 14, "bold"),
        bg=current_theme["btn"],
        fg=current_theme["text"],
        relief="flat",
        width=30,
        height=2,
        command=comando,
        cursor="hand2"
    )
    btn.pack(pady=8)
    btn.bind("<Enter>", hover_on)
    btn.bind("<Leave>", hover_off)
    botones.append(btn)

crear_boton("📸 Registrar nueva seña", lambda: ejecutar(CAPTURA))
crear_boton("👁️ Detectar seña", lambda: ejecutar(RECONOCER))
crear_boton("🧠 Entrenar IA", lambda: ejecutar(ENTRENAR))
crear_boton("📊 Ver señas registradas", lambda: ejecutar(TABLA))  # NUEVO
crear_boton("🌙 Cambiar modo claro / oscuro", toggle_theme)
crear_boton("❌ Salir", root.quit)

# ================= FOOTER =================
footer = tk.Label(
    root,
    text="Proyecto LSM • MediaPipe + IA • 2 Manos",
    bg=current_theme["bg"],
    fg="white",
    font=("Segoe UI", 9)
)
footer.pack(side="bottom", pady=5)

aplicar_tema()

root.mainloop()
