import numpy as np
import os
import shutil
import requests
import json 
import tkinter as tk
from tkinter import messagebox, scrolledtext
from io import BytesIO
from bs4 import BeautifulSoup
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import pyttsx3 
import threading 

# ===============================================================
# --- 1. CONFIGURACIÓN GLOBAL Y FUNCIONES CORE ---
# ===============================================================

detector = MTCNN()

# La inicialización del motor de audio se hará dentro del hilo para aislar recursos.
engine = None 

MODELO_FILENAME = "modelo_deteccion_rostros.h5"
CLASSES_FILENAME = "clase_indices.json"
INPUT_DIR = "./train/"
OUTPUT_DIR = "./dataset_rostros/" 
UMBRAL_CONF = 0.90
CLASE_NO_FAMILIAR = 'No familiar'
LOGO_FILENAME = "VisualSupportLOGO.jpeg" 


def crear_dataset_rostros(input_dir, output_dir):
    """Procesa imágenes en INPUT_DIR y guarda rostros detectados en OUTPUT_DIR."""
    if not os.path.isdir(input_dir):
        return False
        
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    for clase in os.listdir(input_dir):
        clase_path = os.path.join(input_dir, clase)
        if not os.path.isdir(clase_path): continue

        output_class_dir = os.path.join(output_dir, clase)
        os.makedirs(output_class_dir, exist_ok=True)

        for img_name in os.listdir(clase_path):
            img_path = os.path.join(clase_path, img_name)
            try:
                MAX_SIZE = 1000
                img_pil = Image.open(img_path)
                
                if max(img_pil.size) > MAX_SIZE:
                    img_pil.thumbnail((MAX_SIZE, MAX_SIZE))
                    
                if img_pil.mode in ('RGBA', 'P', 'L', 'CMYK'):
                    img_pil = img_pil.convert('RGB')
                    
                img = np.array(img_pil) 
                detecciones = detector.detect_faces(img)
                if len(detecciones) == 0: continue

                x1, y1, w, h = detecciones[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + w, y1 + h

                rostro = img[y1:y2, x1:x2]
                rostro = Image.fromarray(rostro).resize((150, 150))
                rostro.save(os.path.join(output_class_dir, img_name))

            except Exception as e:
                print(f"❌ Error procesando {img_name}: {e}")
                
    return True

def es_url_imagen(url):
    extensiones = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
    return url.lower().endswith(extensiones)

def extraer_imagen_wikimedia(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    candidates = soup.find_all("img")
    for tag in candidates:
        src = tag.get("src")
        if src and ("upload.wikimedia.org" in src or src.startswith("//upload")):
            if src.startswith("//"): src = "https:" + src
            return src
    raise ValueError("❌ No se pudo extraer la imagen de Wikimedia.")

def cargar_imagen_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    if es_url_imagen(url):
        response = requests.get(url, headers=headers)
        return np.array(Image.open(BytesIO(response.content)).convert("RGB"))
    if "commons.wikimedia.org/wiki/" in url:
        url_img = extraer_imagen_wikimedia(url)
        response = requests.get(url_img, headers=headers)
        return np.array(Image.open(BytesIO(response.content)).convert("RGB"))
    raise ValueError("❌ La URL no es una imagen directa ni página compatible.")


# ===============================================================
# --- 3. CLASE DE LA INTERFAZ (Tkinter) ---
# ===============================================================

class DeteccionRostrosApp:
    def __init__(self, master):
        self.master = master
        master.title("Sistema de Detección de Rostros - VisualSupport")
        
        # --- Configuración para Adaptarse a la Pantalla ---
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        initial_width = int(screen_width * 0.8)
        initial_height = int(screen_height * 0.9)
        master.geometry(f"{initial_width}x{initial_height}")
        master.minsize(800, 600) 
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)
        
        # Variables de estado del modelo y resultados
        self.modelo = None
        self.class_indices = {}
        self.num_clases = 0
        self.img_tk_ref = None
        self.logo_tk_ref = None
        self.last_speech_message = "Aún no se ha realizado la primera predicción. Por favor, cargue un modelo y analice una imagen."
        
        # Cargar fuente para dibujar en la imagen
        self.font = None
        try:
            self.font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            self.font = ImageFont.load_default() 

        # Cargar la imagen del logo
        self.logo_original = self.cargar_logo(LOGO_FILENAME)

        self.setup_ui()
        
        # Binding para redimensionar el logo cuando la ventana cambia
        self.master.bind('<Configure>', self.redimensionar_logo_en_evento)


    def cargar_logo(self, filename):
        """Carga la imagen original del logo o devuelve None si falla."""
        if os.path.exists(filename):
            try:
                return Image.open(filename)
            except Exception as e:
                self.log(f"Error cargando logo: {e}", tag="ERROR_LOGO")
                return None
        else:
            self.log(f"Advertencia: Archivo de logo '{filename}' no encontrado.", tag="ALERTA_LOGO")
            return None

    def redimensionar_logo_en_evento(self, event):
        """Redimensiona el logo para que quepa en el ancho de la ventana."""
        # Esta función es llamada SOLO por el evento <Configure> de Tkinter
        if event.widget == self.master and self.logo_original:
            new_width = event.width - 40 # 40px de margen
            self.mostrar_logo(new_width)

    def mostrar_logo(self, max_width):
        """Redimensiona y muestra el logo en la etiqueta."""
        if self.logo_original:
            # Definir tamaño máximo del logo (ej: no más de 300px de ancho)
            MAX_LOGO_WIDTH = 100
            
            target_width = min(max_width, MAX_LOGO_WIDTH)
            
            # Redimensionar manteniendo el aspecto
            logo_width, logo_height = self.logo_original.size
            if logo_width > target_width:
                ratio = target_width / logo_width
                new_size = (target_width, int(logo_height * ratio))
                logo_resized = self.logo_original.resize(new_size, Image.Resampling.LANCZOS)
            else:
                logo_resized = self.logo_original

            self.logo_tk_ref = ImageTk.PhotoImage(logo_resized)
            self.logo_label.config(image=self.logo_tk_ref)
            self.logo_label.image = self.logo_tk_ref
        
    # --- Método de Carga/Entrenamiento (Core) ---
    def cargar_o_entrenar_modelo(self, load_only=False):
        if load_only and os.path.exists(MODELO_FILENAME) and os.path.exists(CLASSES_FILENAME):
            try:
                self.log(f"Cargando modelo entrenado desde: {MODELO_FILENAME}", tag="CARGA")
                self.modelo = load_model(MODELO_FILENAME)
                with open(CLASSES_FILENAME, 'r') as f:
                    self.class_indices = json.load(f)
                self.num_clases = len(self.class_indices)
                return True, "Cargado", self.num_clases
            except Exception as e:
                return False, f"Error al cargar modelo: {e}", 0

        self.log("Iniciando Proceso de Entrenamiento...", tag="INICIO")

        if not crear_dataset_rostros(INPUT_DIR, OUTPUT_DIR):
            return False, "Error al crear dataset. Revise la carpeta './train/'.", 0
        
        train_datagen = ImageDataGenerator(
            rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True, validation_split=0.2
        )
        try:
            train_gen = train_datagen.flow_from_directory(
                OUTPUT_DIR, target_size=(150, 150), batch_size=32, class_mode='categorical', subset="training"
            )
            val_gen = train_datagen.flow_from_directory(
                OUTPUT_DIR, target_size=(150, 150), batch_size=32, class_mode='categorical', subset="validation"
            )
        except Exception as e:
             return False, f"Error en generadores de datos: {e}. ¿Hay al menos 2 clases con imágenes?", 0

        self.num_clases = len(train_gen.class_indices)
        self.class_indices = train_gen.class_indices
        
        if self.num_clases < 2:
            return False, "Error: Se requieren al menos 2 clases para entrenar.", 0

        modelo_cnn = Sequential([
            Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)), MaxPooling2D(2,2),
            Conv2D(64,(3,3),activation='relu'), MaxPooling2D(2,2),
            Conv2D(128,(3,3),activation='relu'), MaxPooling2D(2,2),
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)), Dropout(0.5),
            Dense(self.num_clases, activation='softmax') 
        ])
        modelo_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.modelo = modelo_cnn

        callbacks = [EarlyStopping(patience=50, restore_best_weights=True), ReduceLROnPlateau(factor=0.5, patience=15)]
        
        self.log(f"Comenzando entrenamiento con {self.num_clases} clases...", tag="ENTRENANDO")
        modelo_cnn.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=callbacks, verbose=1)
        
        # El UserWarning sobre el formato HDF5 se mantiene (es una advertencia de Keras)
        modelo_cnn.save(MODELO_FILENAME)
        with open(CLASSES_FILENAME, 'w') as f:
            json.dump(self.class_indices, f)
            
        return True, "Entrenamiento Completo", self.num_clases

    # --- Método de Predicción (Core) ---
    def detectar_y_clasificar(self, img_source):
        if self.modelo is None:
            return "ERROR", "Modelo no cargado. Presione 'Cargar/Entrenar'.", None

        try:
            if img_source.startswith("http"):
                img = cargar_imagen_url(img_source)
            else:
                img = plt.imread(img_source)

            img_draw = Image.fromarray(img).convert("RGB")
            rostros_detectados = []
            detecciones = detector.detect_faces(img)

            if len(detecciones) == 0:
                return "RESULTADO", "❌ No se detectaron rostros.", img_draw

            idx_to_class = {v: k for k, v in self.class_indices.items()}

            for det in detecciones:
                x1, y1, w, h = det['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + w, y1 + h
                
                cara = img[y1:y2, x1:x2]
                rostro_resized = Image.fromarray(cara).resize((150, 150))
                img_array = img_to_array(rostro_resized) / 255.0
                img_array = np.expand_dims(img_array, 0)

                predicciones = self.modelo.predict(img_array, verbose=0)[0]
                predicted_index = np.argmax(predicciones) 
                predicted_class_name = idx_to_class.get(predicted_index, "ERROR_CLASE") 
                confidence = predicciones[predicted_index]
                
                # Lógica de UMBRAL (Se mantiene para clasificar final_class)
                if predicted_class_name != CLASE_NO_FAMILIAR and confidence < UMBRAL_CONF:
                    final_class = CLASE_NO_FAMILIAR.upper()
                    log_message = f"CLASE FORZADA: {final_class} | Confianza: {confidence:.4f} (Original: {predicted_class_name.upper()})"
                else:
                    final_class = predicted_class_name.upper()
                    log_message = f"CLASE: {final_class} | Confianza: {confidence:.4f}"
                    
                rostros_detectados.append({
                    'box': (x1, y1, x2, y2),
                    'clase': final_class,
                    'log': log_message 
                })

            return "OK", rostros_detectados, img_draw

        except Exception as e:
            return "ERROR", f"❌ Error en la clasificación: {e}", None

    # --- Métodos de la UI ---
    
    def iniciar_modelo(self, load_only):
        self.log(f"Iniciando en modo: {'Cargar' if load_only else 'Entrenar'}", tag="INICIO")
        
        success, message, num_classes = self.cargar_o_entrenar_modelo(load_only)
        self.num_clases = num_classes

        if success:
            self.log(f"Operación Exitosa: {message}. Clases encontradas: {self.num_clases}", tag="ÉXITO")
            self.update_clases_display()
            
            # --- CORRECCIÓN DEL ERROR ---
            self.master.update_idletasks()
            # Llamar directamente a mostrar_logo con el ancho actual, NO a redimensionar_logo_en_evento
            self.mostrar_logo(self.master.winfo_width()) 
            # ---------------------------
            
        else:
            messagebox.showerror("Error de Modelo", message)
            self.log(f"Fallo de Operación: {message}", tag="ERROR")
            self.status_label.config(text="Estado: Error", fg='red')
            
    # --- MÉTODO DE AUDIO ASÍNCRONO ---
    def vocalizar_prediccion(self):
        """Vocaliza el último resultado de la predicción en un hilo separado."""
        if not self.last_speech_message or not pyttsx3:
            self.log("Error: Motor de audio no disponible o mensaje vacío.", tag="ERROR_AUDIO")
            return

        def run_audio(text_to_speak):
            try:
                # Inicializar el motor en el hilo local (clave para evitar el bloqueo)
                local_engine = pyttsx3.init()
                local_engine.setProperty('rate', 150)
                
                local_engine.say(text_to_speak)
                local_engine.runAndWait() 
                
                local_engine.stop()
                
                self.log(f"Vocalización completada.", tag="AUDIO")
            except Exception as e:
                self.log(f"Error en hilo de audio: {e}. Intente reinstalar pyttsx3 y dependencias.", tag="ERROR_AUDIO")
        
        # Iniciar el proceso de audio en un nuevo hilo (pasando el mensaje actual)
        self.log(f"Iniciando vocalización: '{self.last_speech_message}' ", tag="AUDIO")
        threading.Thread(target=run_audio, args=(self.last_speech_message,)).start()


    def setup_ui(self):
        main_frame = tk.Frame(self.master, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        # --- CONTENEDOR DEL LOGO Y TÍTULO ---
        header_frame = tk.Frame(main_frame)
        header_frame.pack(side='top', fill='x', pady=5)
        
        # LOGO (Centrado)
        self.logo_label = tk.Label(header_frame)
        self.logo_label.pack(pady=(0, 5))
        self.mostrar_logo(300) 
        
        # TÍTULO RESTAURADO (Centrado)
        tk.Label(header_frame, text="SISTEMA DE DETECCIÓN DE ROSTROS FAMILIARES O NO FAMILIARES", font=("Helvetica", 14, "bold")).pack(pady=(0, 10))

        # --- CONTROL Y ESTADO ---
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill='x', pady=5)
        
        tk.Label(control_frame, text="Modo de Operación:", font=("Helvetica", 10)).pack(side='left', padx=5)
        
        btn_train = tk.Button(control_frame, text="1. Entrenar y Predecir", bg='#4CAF50', fg='white', 
                              command=lambda: self.iniciar_modelo(load_only=False))
        btn_train.pack(side='left', padx=5)

        btn_load = tk.Button(control_frame, text="2. Solo Cargar Modelo", bg='#2196F3', fg='white', 
                             command=lambda: self.iniciar_modelo(load_only=True))
        btn_load.pack(side='left', padx=5)
        
        self.status_label = tk.Label(control_frame, text="Estado: No iniciado", fg='red', font=("Helvetica", 10, "bold"))
        self.status_label.pack(side='right', padx=5)

        info_frame = tk.Frame(main_frame, bd=1, relief=tk.SUNKEN, padx=5, pady=5)
        info_frame.pack(fill='x', pady=10)
        
        tk.Label(info_frame, text="Clases Entrenadas:", font=("Helvetica", 10, "bold")).pack(anchor='w')
        self.clases_text = scrolledtext.ScrolledText(info_frame, height=5, wrap=tk.WORD, state=tk.DISABLED)
        self.clases_text.pack(fill='x', pady=5)

        tk.Label(info_frame, text="Registro de Eventos:", font=("Helvetica", 10, "bold")).pack(anchor='w')
        self.log_text = scrolledtext.ScrolledText(info_frame, height=5, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill='x', pady=5)
        
        self.image_label = tk.Label(main_frame, text="Esperando imagen...")
        self.image_label.pack(fill='both', expand=True, pady=10)
        
        # --- Contenedor de Resultados y Audio ---
        result_audio_frame = tk.Frame(main_frame)
        result_audio_frame.pack(fill='x', pady=5)
        
        self.result_display_label = tk.Label(result_audio_frame, text="Seleccione Modo de Operación para comenzar.", font=("Helvetica", 14, "bold"))
        self.result_display_label.pack(side='left', padx=20) 
        
        btn_audio = tk.Button(result_audio_frame, text="🔊 Escuchar Predicción", bg='#FF5722', fg='white', 
                              command=self.vocalizar_prediccion)
        btn_audio.pack(side='right', padx=20) 

        # --- ENTRADA DE URL (Siempre visible y lista para análisis) ---
        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill='x', pady=5)
        
        tk.Label(input_frame, text="Ruta Local o URL:", font=("Helvetica", 10)).pack(side='left', padx=5)
        self.url_entry = tk.Entry(input_frame, width=50)
        self.url_entry.pack(side='left', fill='x', expand=True, padx=5)
        
        btn_analyze = tk.Button(input_frame, text="Analizar Imagen", bg='#FF9800', fg='white', command=self.analizar_imagen)
        btn_analyze.pack(side='left', padx=5)

    def log(self, message, tag="INFO"):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{tag}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_clases_display(self):
        self.clases_text.config(state=tk.NORMAL)
        self.clases_text.delete('1.0', tk.END)
        if self.class_indices:
            nombres_clase = sorted(list(self.class_indices.keys()))
            self.clases_text.insert(tk.END, ", ".join(nombres_clase))
            self.status_label.config(text=f"Estado: Modelo listo ({self.num_clases} clases)", fg='green')
        else:
            self.clases_text.insert(tk.END, "Ninguna clase cargada.")
            self.status_label.config(text="Estado: ERROR (No hay clases)", fg='red')
        self.clases_text.config(state=tk.DISABLED)

    def analizar_imagen(self):
        self.result_display_label.config(text="")
        self.image_label.config(text="Procesando...")
        
        if self.modelo is None:
            messagebox.showwarning("Advertencia", "El modelo no está cargado. Por favor, cargue o entrene el modelo primero.")
            self.log("Intento de análisis sin modelo cargado.", tag="ALERTA")
            self.image_label.config(text="Esperando imagen...")
            return

        img_source = self.url_entry.get().strip()
        if not img_source:
            messagebox.showwarning("Advertencia", "Por favor, ingrese una ruta o URL.")
            self.image_label.config(text="Esperando imagen...")
            return

        self.log(f"Iniciando análisis de imagen.", tag="ANÁLISIS")
        
        status, result, img_pil = self.detectar_y_clasificar(img_source)
        
        if status == "ERROR":
            messagebox.showerror("Error de Análisis", result)
            self.log(result, tag="ERROR")
            self.image_label.config(text=f"ERROR: {result}")
            # --- MENSAJE FINAL CONSOLIDADO ---
            self.log("Análisis de imagen concluido con error.", tag="FINALIZADO")
            return
        
        # --- Actualización Visual (antes del audio) ---
        if status == "RESULTADO":
            self.log(result, tag="RESULTADO")
            self.image_label.config(text=result)
            self.result_display_label.config(text="ANÁLISIS COMPLETO", fg='blue')
            self.last_speech_message = result 
            self.mostrar_imagen_con_detecciones(img_pil, []) 
            self.vocalizar_prediccion()
            
            # --- MENSAJE FINAL CONSOLIDADO ---
            self.log("Análisis de imagen concluido.", tag="FINALIZADO")
            return

        if status == "OK":
            self.mostrar_imagen_con_detecciones(img_pil, result)
            self.mostrar_resultados_clasificados(result)
            
            # --- VOCALIZACIÓN AUTOMÁTICA DESPUÉS DE MOSTRAR LA IMAGEN Y RESULTADOS ---
            self.vocalizar_prediccion() 
            
            # --- MENSAJE FINAL CONSOLIDADO ---
            self.log("Análisis de imagen concluido.", tag="FINALIZADO")


    def mostrar_resultados_clasificados(self, detections):
        CLASE_NO_FAMILIAR = 'NO FAMILIAR'
        visual_messages = []
        speech_messages = []
        is_known_count = 0

        for i, det in enumerate(detections):
            clase = det['clase']
            
            if clase != CLASE_NO_FAMILIAR:
                is_known_count += 1
                speech_label = "conocida"
            else:
                speech_label = "desconocida"
            
            # Formato visual mejorado: Rostro N: PREDICCIÓN
            visual_message = f"Rostro {i+1}: {clase}"
            visual_messages.append((visual_message))
            
            # Formato para vocalización
            speech_messages.append(f"Rostro número {i+1} identificado como {clase}, persona {speech_label}.")

            # El log detallado se suprimió aquí para simplificar el Registro de Eventos

        # --- Configuración Visual ---
        final_text = " | ".join(visual_messages)
        final_color = 'green' if is_known_count > 0 else 'red'
        
        self.result_display_label.config(text=f"PREDICCIÓN: {final_text}", fg=final_color)
        
        # --- Configuración de Audio (Speech) ---
        num_rostros = len(detections)
        
        if num_rostros == 1:
            audio_intro = f"Análisis completado. Se detectó un rostro."
        else:
            audio_intro = f"Análisis completado. Se detectaron {num_rostros} rostros."
             
        self.last_speech_message = f"{audio_intro}. Detalles: {' '.join(speech_messages)}"


    def mostrar_imagen_con_detecciones(self, img_pil, detections):
        CLASE_NO_FAMILIAR = 'NO FAMILIAR'
        draw = ImageDraw.Draw(img_pil)
        
        if isinstance(detections, list):
            for det in detections:
                box = det['box']
                clase = det['clase']
                
                outline_color = 'green' if clase != CLASE_NO_FAMILIAR else 'red'
                
                draw.rectangle(box, outline=outline_color, width=4)
                
                draw.text((box[0] + 5, box[1] - 20), clase, fill=outline_color, font=self.font)

        # --- TAMAÑO REDUCIDO A 400x300 ---
        ancho_max = 400
        alto_max = 300
        img_pil.thumbnail((ancho_max, alto_max))
        
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.img_tk_ref = img_tk 
        
        self.image_label.config(image=self.img_tk_ref)
        self.image_label.image = self.img_tk_ref 

# ===============================================================
# --- 4. EJECUCIÓN ---
# ===============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = DeteccionRostrosApp(root)
    root.mainloop()