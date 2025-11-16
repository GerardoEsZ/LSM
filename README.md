# LSM – Reconocimiento de Lengua de Señas Mexicana con IA

Proyecto de reconocimiento de señas usando **MediaPipe**, **TensorFlow** y **LSTM**, capaz de:

- Detectar **dos manos** (izquierda y derecha).
- Reconocer **gestos con movimiento** (secuencias de frames).
- Funcionar en dos modos:
  - **Modo letras** → deletreo (A, B, C, Z, etc.).
  - **Modo palabras** → gestos completos (HOLA, SI, NO, …).

---

## 1. Estructura del proyecto

Archivos principales:

- `capturar_secuencias_dos_manos.py`  
  Captura secuencias de video de los gestos (con movimiento y dos manos) y guarda:
  - `gestures_sequences.npz` → dataset de secuencias.
  - `gestures_labels.json` → mapa de índice → nombre de clase (letra/palabra).

- `entrenar_lstm_gestos.py`  
  Entrena un modelo LSTM con las secuencias capturadas y guarda:
  - `modelo_gestos_lstm.h5` → modelo entrenado.

- `reconocer_gestos_modos.py`  
  Usa la cámara web para reconocer señas en tiempo real en dos modos:
  - **LETRAS** → solo clases de 1 carácter.
  - **PALABRAS** → solo clases con más de 1 carácter.

Archivos generados (NO se suben a GitHub):

- `gestures_sequences.npz` → dataset de entrenamiento.
- `gestures_labels.json` → nombres de las clases.
- `modelo_gestos_lstm.h5` → modelo entrenado (puede ser grande).

---

## 2. Instalación

### 2.1. Clonar el repositorio

```bash
git clone <URL_DE_TU_REPO>
cd <NOMBRE_DEL_REPO>
