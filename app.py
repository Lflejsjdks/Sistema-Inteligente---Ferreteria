import cv2
import numpy as np
import json
import os
import time
import webbrowser
from threading import Timer
from flask import Flask, render_template, Response, jsonify
# AGREGADO: Importación de Red Neuronal YOLO
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.json")
CAPTURAS_DIR = os.path.join(BASE_DIR, "capturas")

if not os.path.exists(CAPTURAS_DIR):
    os.makedirs(CAPTURAS_DIR)

# --- VARIABLES GLOBALES ---
camera = cv2.VideoCapture(1) 
modelo_yolo = YOLO("yolov8n.pt") 
last_save_time = 0
productos_detectados = set() 

def actualizar_stock_json(nombre_producto):
    try:
        if not os.path.exists(DATA_PATH): return
        with open(DATA_PATH, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            modificado = False
            for item in data:
                if item['nombre'].lower() in nombre_producto.lower() and item['cantidad'] > 0:
                    item['cantidad'] -= 1
                    modificado = True
                    break 
            if modificado:
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
                print(f"✅ Stock actualizado para: {nombre_producto}")
    except Exception as e:
        print(f"❌ Error al escribir en JSON: {e}")

def procesar_vision(frame):
    global last_save_time, productos_detectados
    
    # --- 1. DETECCIÓN POR FORMA (YOLO) ---
    resultados = modelo_yolo(frame, conf=0.5, verbose=False)
    mapeo_formas = {
        "scissors": "Tijeras Amarillas",
        "pliers": "Alicate Universal",
        "hammer": "Martillo de Uña"
    }

    encontrado_por_ia = False

    for r in resultados:
        for box in r.boxes:
            cls = int(box.cls[0])
            label_ingles = modelo_yolo.names[cls]
            
            if label_ingles in mapeo_formas:
                nombre_real = mapeo_formas[label_ingles]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Dibujamos basado en la forma detectada por IA
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{nombre_real.upper()} (IA)", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                if nombre_real not in productos_detectados:
                    productos_detectados.add(nombre_real)
                    actualizar_stock_json(nombre_real)
                
                encontrado_por_ia = True
                break 
        if encontrado_por_ia: break 

    # --- 2. LÓGICA DE RESPALDO (COLOR + FILTRO DE ÁREA Y PROPORCIÓN) ---
    if not encontrado_por_ia:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        amarillo_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        azul_mask = cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))
        gris_mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 50, 200]))

        detecciones = [
            (amarillo_mask, "Tijeras Amarillas", (0, 255, 255), 0.5), # Relación ancho/alto
            (azul_mask, "Alicate Universal", (255, 0, 0), 0.5),
            (gris_mask, "Martillo de Uña", (169, 169, 169), 0.8) # El martillo suele ser más largo
        ]

        kernel = np.ones((5, 5), np.uint8)

        for mask, nombre, color_bgr, ratio_min in detecciones:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contornos:
                # INTEGRACIÓN: Área subida a 15000 para ignorar el fondo de la pared
                if cv2.contourArea(c) > 15000: 
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # INTEGRACIÓN: Filtro de proporción para asegurar que es un objeto real
                    if float(h)/w > ratio_min:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 2)
                        cv2.putText(frame, f"{nombre.upper()} DETECTADO", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
                        
                        if nombre not in productos_detectados:
                            productos_detectados.add(nombre)
                            actualizar_stock_json(nombre)
                        
                        return frame 
            
    return frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success: break
        frame = procesar_vision(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stock')
def get_stock():
    global productos_detectados
    try:
        if productos_detectados and os.path.exists(DATA_PATH):
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                todo_el_stock = json.load(f)
            data_filtrada = [p for p in todo_el_stock if p['nombre'] in productos_detectados]
            return jsonify(data_filtrada)
        return jsonify([]) 
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)