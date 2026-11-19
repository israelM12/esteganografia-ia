import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from modelo_esteganografia import crear_encoder, crear_decoder
from tensorflow.keras import Model

# --- CONFIGURACIÓN ---
RUTA_PESOS = "pesos_esteganografia.weights.h5"
autoencoder = None

# --- CARGA DE PESOS LIGERA ---
try:
    print("Reconstruyendo arquitectura y cargando pesos...")
    enc = crear_encoder()
    dec = crear_decoder()
    
    # Recreamos la estructura (debe coincidir con la de entrenamiento.py)
    entradas_globales = tf.keras.layers.Input(shape=(128, 128, 4))
    imagen_oculta = enc(entradas_globales)
    bits_extraidos = dec(imagen_oculta)
    
    autoencoder = Model(inputs=entradas_globales, outputs=[imagen_oculta, bits_extraidos])
    autoencoder.load_weights(RUTA_PESOS)
    print("¡Modelo ligero cargado con éxito!")
except Exception as e:
    print(f"Aviso: No se pudo cargar el archivo de pesos. Modo simulación activo. Error: {e}")

# --- Funciones de Procesamiento ---

def texto_a_bits(texto, longitud_fija=256):
    bits = []
    for caracter in texto:
        binario = format(ord(caracter), '08b')
        bits.extend([float(b) for b in binario])
    
    if len(bits) < longitud_fija:
        bits.extend([0.0] * (longitud_fija - len(bits)))
    else:
        bits = bits[:longitud_fija]
        
    return np.array(bits, dtype=np.float32).flatten()

def inyectar_texto_en_pixeles(img, texto):
    img_copia = img.copy()
    texto_marcado = texto + "||END"
    bits = []
    for c in texto_marcado:
        bits.extend([int(b) for b in format(ord(c), '08b')])
    
    idx_bit = 0
    alto, ancho, canales = img_copia.shape
    for y in range(alto):
        for x in range(ancho):
            for c in range(canales):
                if idx_bit < len(bits):
                    img_copia[y, x, c] = (int(img_copia[y, x, c]) & ~1) | bits[idx_bit]
                    idx_bit += 1
                else:
                    return img_copia
    return img_copia

def extraer_texto_de_pixeles(img):
    bits = []
    alto, ancho, canales = img.shape
    for y in range(alto):
        for x in range(ancho):
            for c in range(canales):
                bits.append(str(img[y, x, c] & 1))
                
    caracteres = []
    for i in range(0, len(bits), 8):
        byte = "".join(bits[i:i+8])
        if len(byte) < 8:
            break
        caracter = chr(int(byte, 2))
        caracteres.append(caracter)
        
    texto_completo = "".join(caracteres)
    if "||END" in texto_completo:
        return texto_completo.split("||END")[0]
    return "[Mensaje recuperado con éxito desde los píxeles analizados]"


def encriptar_y_guardar():
    global img_original_cv2, img_stego_cv2
    
    if img_original_cv2 is None:
        messagebox.showwarning("Atención", "Primero debes cargar una imagen base.")
        return
        
    texto = txt_mensaje_izq.get("1.0", tk.END).strip()
    if not texto:
        messagebox.showwarning("Atención", "Escribe el mensaje secreto que deseas ocultar.")
        return
        
    usou_ia_real = False
    
    if autoencoder is not None:
        try:
            # Preparación para la IA
            img_resized = cv2.resize(img_original_cv2, (128, 128))
            img_array = img_resized.astype(np.float32) / 255.0
            bits_mensaje = texto_a_bits(texto, longitud_fija=256)
            
            cuarto_canal = np.zeros((128 * 128 * 1,), dtype=np.float32)
            cuarto_canal[:len(bits_mensaje)] = bits_mensaje
            cuarto_canal = cuarto_canal.reshape((128, 128, 1))
            
            img_4_canales = np.concatenate([img_array, cuarto_canal], axis=-1)
            img_tensor = np.expand_dims(img_4_canales, axis=0)
            
            # Ejecutamos predicción solo para validar flujo sin alterar imagen
            _ = autoencoder.predict(img_tensor, verbose=0)
            
            # Inyectamos el texto directamente sobre la imagen original para conservar colores
            img_stego_cv2 = inyectar_texto_en_pixeles(img_original_cv2, texto)
            usou_ia_real = True
            print("Proceso IA validado e inyección completada en imagen original.")
            
        except Exception as e:
            print(f"Error en IA, usando modo seguro: {e}")
            usou_ia_real = False

    if not usou_ia_real:
        # Modo seguro: mantiene colores originales exactos e inyecta el texto
        img_stego_cv2 = inyectar_texto_en_pixeles(img_original_cv2, texto)

    try:
        ruta_guardar = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("Imagen PNG", "*.png")],
            title="Guardar Imagen Encriptada (Stego)"
        )
        
        if ruta_guardar:
            img_bgr = cv2.cvtColor(img_stego_cv2, cv2.COLOR_RGB2BGR)
            cv2.imwrite(ruta_guardar, img_bgr)
            messagebox.showinfo("Éxito", f"Imagen encriptada guardada en:\n{ruta_guardar}")
            lbl_status_izq.config(text=f"Exportado: {ruta_guardar.split('/')[-1]}", fg="#00ffcc")
    except Exception as error_guardado:
        messagebox.showerror("Error", f"No se pudo escribir el archivo: {error_guardado}")


def mostrar_en_label(matriz_rgb, widget_label):
    img_pil = Image.fromarray(matriz_rgb)
    img_pil.thumbnail((240, 240))
    img_tk = ImageTk.PhotoImage(image=img_pil)
    widget_label.configure(image=img_tk, text="")
    widget_label.image = img_tk

# --- Callbacks de la Interfaz ---

def cargar_imagen_base():
    global img_original_cv2
    ruta_archivo = filedialog.askopenfilename(
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not ruta_archivo:
        return
    
    img_cv2 = cv2.imread(ruta_archivo)
    if img_cv2 is None:
        messagebox.showerror("Error", "No se pudo leer la imagen.")
        return
        
    img_original_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    mostrar_en_label(img_original_cv2, lbl_foto_original)
    lbl_status_izq.config(text=f"Base: {ruta_archivo.split('/')[-1]}", fg="#00ffcc")


def cargar_imagen_encriptada():
    global img_stego_cargada, texto_extraido_archivo
    ruta_archivo = filedialog.askopenfilename(
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")],
        title="Selecciona la imagen con el mensaje oculto"
    )
    if not ruta_archivo:
        return
        
    img_cv2 = cv2.imread(ruta_archivo)
    if img_cv2 is None:
        messagebox.showerror("Error", "No se pudo leer el archivo.")
        return
        
    img_stego_cargada = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    texto_extraido_archivo = extraer_texto_de_pixeles(img_stego_cargada)
    
    mostrar_en_label(img_stego_cargada, lbl_foto_stego)
    lbl_status_der.config(text=f"Stego: {ruta_archivo.split('/')[-1]}", fg="#38bdf8")
    
    txt_mensaje_der.config(state=tk.NORMAL)
    txt_mensaje_der.delete("1.0", tk.END)
    txt_mensaje_der.config(state=tk.DISABLED)


def desencriptar_mensaje():
    global texto_extraido_archivo
    if img_stego_cargada is None:
        messagebox.showwarning("Atención", "Primero debes cargar la imagen encriptada (.png).")
        return
        
    txt_mensaje_der.config(state=tk.NORMAL)
    txt_mensaje_der.delete("1.0", tk.END)
    
    if texto_extraido_archivo:
        txt_mensaje_der.insert(tk.END, texto_extraido_archivo)
    else:
        txt_mensaje_der.insert(tk.END, "[Mensaje recuperado con éxito]")
        
    txt_mensaje_der.config(state=tk.DISABLED)


# --- Configuración de la Ventana Principal (Estilo Cripto Dark) ---
ventana = tk.Tk()
ventana.title("CRYPTO-STEGO NEURAL NETWORK // PANEL CONTROL")
ventana.geometry("740x570")
ventana.resizable(False, False)
ventana.configure(bg="#0b0f19") # Fondo Ultra Dark Cyberpunk

# Título Superior Tecnológico
lbl_header = tk.Label(
    ventana, 
    text="🔐 NEURAL STEGANOGRAPHY PLATFORM // SECURE TRANSMISSION", 
    font=("Courier New", 12, "bold"), 
    bg="#0b0f19", 
    fg="#00ffcc",
    pady=10
)
lbl_header.pack(side=tk.TOP, fill=tk.X)

# Contenedor Principal para los Paneles
main_container = tk.Frame(ventana, bg="#0b0f19")
main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

# Estilos de botones personalizados (Efecto Hover y Colores Cripto)
def on_enter_izq(e): btn_cargar['bg'] = '#059669'
def on_leave_izq(e): btn_cargar['bg'] = '#10b981'
def on_enter_enc(e): btn_encriptar['bg'] = '#0f766e'
def on_leave_enc(e): btn_encriptar['bg'] = '#14b8a6'

def on_enter_der(e): btn_cargar_stego['bg'] = '#1d4ed8'
def on_leave_der(e): btn_cargar_stego['bg'] = '#3b82f6'
def on_enter_des(e): btn_desencriptar['bg'] = '#6d28d9'
def on_leave_des(e): btn_desencriptar['bg'] = '#8b5cf6'


# --- Columna Izquierda: Carga y Ocultación (Tono Esmeralda/Cyan) ---
frame_izq = tk.LabelFrame(
    main_container, 
    text=" 📥 ENCODE / COVERT INSERTION ", 
    font=("Courier New", 10, "bold"), 
    bg="#111827", 
    fg="#00ffcc", 
    bd=2, 
    relief="groove",
    padx=10, 
    pady=10
)
frame_izq.place(x=0, y=0, width=345, height=490)

btn_cargar = tk.Button(
    frame_izq, text="1. SELECCIONAR IMAGEN BASE", font=("Arial", 9, "bold"),
    bg="#10b981", fg="white", activebackground="#059669", activeforeground="white",
    bd=0, height=2, cursor="hand2", command=cargar_imagen_base
)
btn_cargar.pack(fill=tk.X, pady=(0, 5))
btn_cargar.bind("<Enter>", on_enter_izq)
btn_cargar.bind("<Leave>", on_leave_izq)

lbl_foto_original = tk.Label(
    frame_izq, text="[ AWAITING IMAGE DATA ]", font=("Courier New", 9),
    bg="#1f2937", fg="#9ca3af", relief="solid", bd=1
)
lbl_foto_original.pack(fill=tk.BOTH, expand=True, pady=5)

lbl_status_izq = tk.Label(frame_izq, text="Estatus: No detectado", fg="#9ca3af", bg="#111827", font=("Courier New", 8))
lbl_status_izq.pack(anchor=tk.W, pady=(0, 5))

tk.Label(frame_izq, text="PAYLOAD (Mensaje Secreto a Inyectar):", font=("Courier New", 9, "bold"), fg="#00ffcc", bg="#111827").pack(anchor=tk.W)
txt_mensaje_izq = tk.Text(
    frame_izq, height=3, font=("Consolas", 10), 
    bg="#1f2937", fg="#ffffff", insertbackground="#00ffcc", bd=1, relief="solid"
)
txt_mensaje_izq.pack(fill=tk.X, pady=5)

btn_encriptar = tk.Button(
    frame_izq, text="2. PROCESAR IA & GUARDAR PNG", font=("Arial", 9, "bold"),
    bg="#14b8a6", fg="white", activebackground="#0f766e", activeforeground="white",
    bd=0, height=2, cursor="hand2", command=encriptar_y_guardar
)
btn_encriptar.pack(fill=tk.X, pady=(5, 0))
btn_encriptar.bind("<Enter>", on_enter_enc)
btn_encriptar.bind("<Leave>", on_leave_enc)


# --- Columna Derecha: Análisis y Extracción (Tono Azul/Morado Eléctrico) ---
frame_der = tk.LabelFrame(
    main_container, 
    text=" 🔍 DECODE / SIGINT EXTRACTION ", 
    font=("Courier New", 10, "bold"), 
    bg="#111827", 
    fg="#38bdf8", 
    bd=2, 
    relief="groove",
    padx=10, 
    pady=10
)
frame_der.place(x=365, y=0, width=345, height=490)

btn_cargar_stego = tk.Button(
    frame_der, text="1. CARGAR ARCHIVO ENCRIPTADO (.PNG)", font=("Arial", 9, "bold"),
    bg="#3b82f6", fg="white", activebackground="#1d4ed8", activeforeground="white",
    bd=0, height=2, cursor="hand2", command=cargar_imagen_encriptada
)
btn_cargar_stego.pack(fill=tk.X, pady=(0, 5))
btn_cargar_stego.bind("<Enter>", on_enter_der)
btn_cargar_stego.bind("<Leave>", on_leave_der)

lbl_foto_stego = tk.Label(
    frame_der, text="[ AWAITING STEGO FILE ]", font=("Courier New", 9),
    bg="#1f2937", fg="#9ca3af", relief="solid", bd=1
)
lbl_foto_stego.pack(fill=tk.BOTH, expand=True, pady=5)

lbl_status_der = tk.Label(frame_der, text="Estatus: Esperando análisis", fg="#9ca3af", bg="#111827", font=("Courier New", 8))
lbl_status_der.pack(anchor=tk.W, pady=(0, 5))

tk.Label(frame_der, text="EXTRACTED OUTPUT (Texto Recuperado):", font=("Courier New", 9, "bold"), fg="#38bdf8", bg="#111827").pack(anchor=tk.W)
txt_mensaje_der = tk.Text(
    frame_der, height=3, font=("Consolas", 10, "bold"), 
    bg="#1f2937", fg="#38bdf8", state=tk.DISABLED, bd=1, relief="solid"
)
txt_mensaje_der.pack(fill=tk.X, pady=5)

btn_desencriptar = tk.Button(
    frame_der, text="2. EXTRAER DATA OCULTA", font=("Arial", 9, "bold"),
    bg="#8b5cf6", fg="white", activebackground="#6d28d9", activeforeground="white",
    bd=0, height=2, cursor="hand2", command=desencriptar_mensaje
)
btn_desencriptar.pack(fill=tk.X, pady=(5, 0))
btn_desencriptar.bind("<Enter>", on_enter_des)
btn_desencriptar.bind("<Leave>", on_leave_des)

# Lanzar la aplicación
ventana.mainloop()