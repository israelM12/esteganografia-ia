import cv2
import numpy as np
# Importamos las funciones que acabamos de hacer y validar en el paso anterior
from preprocesamiento import texto_a_bits

def preparar_imagen(ruta_imagen, tamano=128):
    """
    Carga una imagen, la redimensiona a 128x128, la pasa a RGB
    y la normaliza en un rango de [0.0, 1.0].
    """
    # 1. Leer la imagen usando OpenCV
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise FileNotFoundError(f"No se pudo encontrar o leer la imagen en: {ruta_imagen}")
    
    # OpenCV lee en BGR por defecto, la convertimos a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Redimensionar de forma eficiente a 128x128
    img_redim = cv2.resize(img_rgb, (tamano, tamano))
    
    # 3. Normalizar a flotantes entre 0 y 1
    img_normalizada = img_redim.astype(np.float32) / 255.0
    
    return img_normalizada

def expandir_mensaje_a_matriz(vector_bits, tamano=128):
    """
    Toma el vector de 256 bits y lo expande matemáticamente 
    para cubrir una matriz de 128x128x1 llena con esos datos.
    """
    # Para rellenar una matriz de 128x128 (16,384 píxeles) con solo 256 bits,
    # repetimos el vector de bits de forma secuencial usando np.resize
    matriz_plana = np.resize(vector_bits, (tamano * tamano))
    
    # Le damos la forma bidimensional con 1 canal: (128, 128, 1)
    matriz_mensaje = matriz_plana.reshape((tamano, tamano, 1))
    
    return matriz_mensaje
# =============================================================================
# PRUEBA DE PREPARACIÓN DE IMAGEN Y TEXTO (4 CANALES)
# =============================================================================
if __name__ == "__main__":
    # 1. Definimos las rutas y el mensaje de prueba
    ruta_test = "C:/Users/jesus/OneDrive/Escritorio/ProyectoIA/test.jpg"
    mensaje_secreto = "Proyecto Esteganografia 10/10"
    
    print("--- Iniciando Combinación de Datos ---")
    
    # 2. Procesamos el texto a bits
    bits = texto_a_bits(mensaje_secreto)
    print(f"Texto listo en bits. Tamaño: {bits.shape}")
    
    # 3. Procesamos la imagen de prueba
    imagen_rgb = preparar_imagen(ruta_test)
    print(f"Imagen cargada y normalizada. Tamaño: {imagen_rgb.shape} (RGB)")
    
    # 4. Expandimos el mensaje a las dimensiones de la imagen
    matriz_texto = expandir_mensaje_a_matriz(bits)
    print(f"Matriz de texto expandida. Tamaño: {matriz_texto.shape} (1 Canal)")
    
    # 5. FUSIÓN DE CANALES: Pegamos la imagen y el texto en el eje de los canales (axis=2)
    # Esto une (128,128,3) con (128,128,1) -> Resultado: (128,128,4)
    super_matriz = np.concatenate([imagen_rgb, matriz_texto], axis=2)
    
    print("\n==================================================")
    print(f"¡ÉXITO! Súper matriz de entrada creada.")
    print(f"Tamaño final para la Red Neuronal: {super_matriz.shape}")
    print("==================================================")
